// ${generated_comment}

#include <Python.h>

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Size.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/autograd/utils/error_messages.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/jit/tracer.h"
#ifdef USE_CUDA
#include "torch/csrc/cuda/Stream.h"
#include "torch/csrc/cuda/Event.h"
#endif
#include "torch/csrc/utils/cuda_lazy_init.h"
#include "torch/csrc/utils/object_ptr.h"
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

#include <ATen/ATen.h>
#include "c10/util/Optional.h"

#include <stdexcept>

using at::DeviceGuard;
using at::device_of;
using at::OptionalDeviceGuard;
using at::Backend;
using at::Scalar;
using at::ScalarType;
using at::Tensor;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

static PyObject * THPVariable__is_view(PyObject *self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
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
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
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
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (jit::tracer::isTracing()) {
      return wrap(jit::tracer::getSizeOf(self_, r.toInt64(0)));
    } else {
      return wrap(self_.size(r.toInt64(0)));
    }
  } else if (r.idx == 1) {
    // we can't do the normal wrapping here because IntArrayRef maps to both
    // torch.Size and tuple in python.
    return THPSize_New(self_);
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
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(self_.stride(r.toInt64(0)));
  } else if (r.idx == 1) {
    // yes, this is called strides in ATen.
    IntArrayRef strides = self_.strides();
    // we can't do the normal wrapping here because IntArrayRef maps to both
    // torch.Size and tuple in python
    return THPUtils_packInt64Array(strides.size(), strides.data());
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
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(self.get_device());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_has_names(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(self.has_names());
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_data_ptr(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(self.data_ptr());
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_storage_offset(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(self.storage_offset());
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_dim(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
   return THPUtils_packInt64(self_.dim());
   END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_numel(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
   return THPUtils_packInt64(self_.numel());
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
  auto r = parser.parse(args, kwargs, parsed_args);
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  auto memory_format = r.memoryformat(0);
  // avoids touching the GIL or current device if self is already contiguous
  if (self_.is_contiguous(memory_format)) {
    // NOTE: this logic is duplicated from VariableType.cpp. Since we need to
    // record this call to contiguous() in the trace regardless of whether
    // we actually call contiguous here, we need to record this information
    // manually.
    if (jit::tracer::isTracing()) {
      auto tracer_state = jit::tracer::getTracingState();
      auto node = tracer_state->graph->create(jit::aten::contiguous, /*num_outputs=*/0);
      jit::tracer::recordSourceLocation(node);
      jit::tracer::addInputs(node, "self", self_);
      jit::tracer::addInputs(node, "memory_format", memory_format);
      tracer_state->graph->insertNode(node);
      jit::tracer::addOutput(node, self_);
    }
    Py_INCREF(self);
    return self;
  }
  return THPVariable_Wrap(dispatch_contiguous(self_, memory_format));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_copy_(Tensor & self, const Tensor & other, bool non_blocking) {
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
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  return THPVariable_Wrap(dispatch_copy_(self_, r.tensor(0), r.toBool(1)));
  END_HANDLE_TH_ERRORS
}

static double dispatch_to_CDouble(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.item<double>();
}

static std::complex<double> dispatch_to_CComplexDouble(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.item<std::complex<double>>();
}

static int64_t dispatch_to_CLong(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.item<int64_t>();
}

static bool dispatch_to_Bool(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.item<bool>();
}

static PyObject * THPVariable_float_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  jit::tracer::warn("Converting a tensor to a Python float", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return wrap(dispatch_to_CDouble(self_));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_integral_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  jit::tracer::warn("Converting a tensor to a Python integer", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
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
  jit::tracer::warn("Converting a tensor to a Python index", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  // TODO: change the condition to `self_.dim() != 0` once we expose scalars
  // in PyTorch.
  if (!isIntegralType(self_.scalar_type(), /*includeBool=*/true) || self_.numel() != 1) {
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
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
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
   auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
   ParsedArgs<1> parsed_args;
   auto r = parser.parse(args, kwargs, parsed_args);
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
    "nonzero()|deprecated",
    "nonzero(*, bool as_tuple=False)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
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
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto device = r.isNone(0) ? at::Device(at::DeviceType::CUDA) : r.device(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  TORCH_CHECK(device.is_cuda(), "Invalid device, must be cuda device");
  torch::utils::cuda_lazy_init();
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_to_type(PyObject* self, ScalarType scalarType, c10::optional<c10::MemoryFormat> optional_memory_format) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return THPVariable_Wrap(dispatch_to(self_, scalarType, false, false, optional_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_byte(PyObject* self, PyObject* args, PyObject* kwargs)  {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "byte(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
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
  auto r = parser.parse(args, kwargs, parsed_args);
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
  auto r = parser.parse(args, kwargs, parsed_args);
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
  auto r = parser.parse(args, kwargs, parsed_args);
  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Float, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_half(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "half(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
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
  auto r = parser.parse(args, kwargs, parsed_args);
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
  auto r = parser.parse(args, kwargs, parsed_args);
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
  auto r = parser.parse(args, kwargs, parsed_args);
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
  auto r = parser.parse(args, kwargs, parsed_args);
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
  auto r = parser.parse(args, kwargs, parsed_args);
  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::BFloat16, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_element_size(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return THPUtils_packInt64(self_.element_size());
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc PyObjects not declarable in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_numpy(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("Converting a tensor to a NumPy array", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return torch::utils::tensor_to_numpy(self_);
  END_HANDLE_TH_ERRORS
}

// TODO: move this to ATen. We would need to expose Stream objects in ATen.
static PyObject * THPVariable_record_stream(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
#ifdef USE_CUDA
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (!THCPStream_Check(arg)) {
    return PyErr_Format(PyExc_TypeError, "expected Stream object");
  }
  c10::cuda::CUDACachingAllocator::recordStream(self_.storage().data_ptr(), at::cuda::CUDAStream::unpack(((THCPStream*)arg)->cdata));
  Py_RETURN_NONE;
#else
  throw std::runtime_error("PyTorch compiled without CUDA support");
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_requires_grad_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "requires_grad_(bool requires_grad=True)",
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto requires_grad = r.toBool(0);
  // should we throw if requires_grad is true?  var.requires_grad = True throws here
  // but it's nice to let this be a no-op.
  if (!self_.is_leaf() && !requires_grad) {
    throw std::runtime_error(autograd::utils::requires_grad_leaf_error(requires_grad));
  }
  if (requires_grad && !self_.is_floating_point()) {
    throw std::runtime_error("only Tensors of floating point dtype can require gradients");
  }
  self_.set_requires_grad(requires_grad);
  return THPVariable_Wrap(self_);
  END_HANDLE_TH_ERRORS
}

inline bool dispatch_is_contiguous(Tensor & self, MemoryFormat memory_format) {
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
  auto r = parser.parse(args, kwargs, parsed_args);
  auto memory_format = r.memoryformat(0);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_is_contiguous(self, memory_format));
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_item(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("Converting a tensor to a Python number", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.is_floating_point()) {
    return wrap(dispatch_to_CDouble(self_));
  } else if (self_.is_complex()) {
    return wrap(dispatch_to_CComplexDouble(self_));
  } else if (self_.scalar_type() == ScalarType::Bool) {
    return wrap(dispatch_to_Bool(self_));
  } else {
    return wrap(dispatch_to_CLong(self_));
  }
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc no support for first class functions in native_functions.yaml
// See: ATen/native/README.md for more context
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

// implemented on the python object bc no support for first class functions in native_functions.yaml
// See: ATen/native/README.md for more context
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
  OptionalDeviceGuard device_guard(device_of(self_));
  return THPVariable_Wrap(torch::utils::legacy_tensor_new(legacyExtractDispatchKey(self_), self_.scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_ones(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  OptionalDeviceGuard device_guard(device_of(self_));
  return THPVariable_Wrap(torch::utils::new_ones(legacyExtractDispatchKey(self_), self_.scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  OptionalDeviceGuard device_guard(device_of(self_));
  return THPVariable_Wrap(torch::utils::new_tensor(legacyExtractDispatchKey(self_), self_.scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_storage(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return createPyObject(self_.storage());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_storage_type(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  auto storage = THPObjectPtr(createPyObject(self_.storage()));
  auto storage_type = (PyObject*)Py_TYPE(storage);
  Py_INCREF(storage_type);
  return storage_type;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_to(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto parsed = parse_to_conversion(args, kwargs, /*allow_copy*/ true);
  auto& device = std::get<0>(parsed);
  auto& scalarType = std::get<1>(parsed);
  auto non_blocking = std::get<2>(parsed);
  auto copy = std::get<3>(parsed);
  auto opt_memory_format = std::get<4>(parsed);
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
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
  jit::tracer::warn("Converting a tensor to a Python list", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto self_ = reinterpret_cast<THPVariable*>(self)->cdata;
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
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
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

// XXX: ops that are bound here are not exposed to the C++ api nor the JIT.
// Any new ops added here should be accompanied with a comment why they are not
// being registered through native_functions.yaml, and be tagged cpp / JIT
PyMethodDef variable_methods[] = {
  // These magic methods are all implemented on python object to wrap NotImplementedError
  {"__add__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_add>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__radd__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_add>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__iadd__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_add_>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__rmul__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_mul>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__mul__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_mul>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__imul__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_mul_>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__sub__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_sub>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__isub__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_sub_>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__div__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_div>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__truediv__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_div>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__idiv__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_div_>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__mod__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_remainder>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__bool__", (PyCFunction)THPVariable_bool_scalar, METH_NOARGS, NULL},
  {"__float__", (PyCFunction)THPVariable_float_scalar, METH_NOARGS, NULL},
  {"__int__", (PyCFunction)THPVariable_integral_scalar, METH_NOARGS, NULL},
  {"__long__", (PyCFunction)THPVariable_integral_scalar, METH_NOARGS, NULL},
  {"__index__", (PyCFunction)THPVariable_index_scalar, METH_NOARGS, NULL},
  {"__nonzero__", (PyCFunction)THPVariable_bool_scalar, METH_NOARGS, NULL},
  {"__invert__", (PyCFunction)THPVariable_invert, METH_NOARGS, NULL},
  {"__matmul__", (PyCFunction)(void(*)(void))TypeError_to_NotImplemented_<THPVariable_matmul>, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_is_view", (PyCFunction)THPVariable__is_view, METH_NOARGS, NULL},
  {"apply_", (PyCFunction)THPVariable_apply_, METH_O, NULL},
  {"bfloat16", (PyCFunction)(void(*)(void))THPVariable_bfloat16, METH_VARARGS | METH_KEYWORDS, NULL},
  {"byte", (PyCFunction)(void(*)(void))THPVariable_byte, METH_VARARGS | METH_KEYWORDS, NULL},
  {"char", (PyCFunction)(void(*)(void))THPVariable_char, METH_VARARGS | METH_KEYWORDS, NULL},
  {"contiguous", (PyCFunction)(void(*)(void))THPVariable_contiguous, METH_VARARGS | METH_KEYWORDS, NULL},
  {"copy_", (PyCFunction)(void(*)(void))THPVariable_copy_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cpu", (PyCFunction)(void(*)(void))THPVariable_cpu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cuda", (PyCFunction)(void(*)(void))THPVariable_cuda, METH_VARARGS | METH_KEYWORDS, NULL},
  {"data_ptr", (PyCFunction)THPVariable_data_ptr, METH_NOARGS, NULL},
  {"dim", (PyCFunction)THPVariable_dim, METH_NOARGS, NULL},
  {"has_names", (PyCFunction)THPVariable_has_names, METH_NOARGS, NULL},
  {"double", (PyCFunction)(void(*)(void))THPVariable_double, METH_VARARGS | METH_KEYWORDS, NULL},
  {"element_size", (PyCFunction)THPVariable_element_size, METH_NOARGS, NULL},
  {"float", (PyCFunction)(void(*)(void))THPVariable_float, METH_VARARGS | METH_KEYWORDS, NULL},
  {"get_device", (PyCFunction)THPVariable_get_device, METH_NOARGS, NULL},
  {"bool", (PyCFunction)(void(*)(void))THPVariable_bool, METH_VARARGS | METH_KEYWORDS, NULL},
  {"half", (PyCFunction)(void(*)(void))THPVariable_half, METH_VARARGS | METH_KEYWORDS, NULL},
  {"int", (PyCFunction)(void(*)(void))THPVariable_int, METH_VARARGS | METH_KEYWORDS, NULL},
  {"is_contiguous", (PyCFunction)(void(*)(void))THPVariable_is_contiguous, METH_VARARGS | METH_KEYWORDS, NULL},
  {"item", (PyCFunction)THPVariable_item, METH_NOARGS, NULL},
  {"long", (PyCFunction)(void(*)(void))THPVariable_long, METH_VARARGS | METH_KEYWORDS, NULL},
  {"map_", (PyCFunction)(void(*)(void))THPVariable_map_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"map2_", (PyCFunction)(void(*)(void))THPVariable_map2_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ndimension", (PyCFunction)THPVariable_dim, METH_NOARGS, NULL},
  {"nelement", (PyCFunction)THPVariable_numel, METH_NOARGS, NULL},
  {"new", (PyCFunction)(void(*)(void))THPVariable_new, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_ones", (PyCFunction)(void(*)(void))THPVariable_new_ones, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_tensor", (PyCFunction)(void(*)(void))THPVariable_new_tensor, METH_VARARGS | METH_KEYWORDS, NULL},
  {"nonzero", (PyCFunction)(void(*)(void))THPVariable_nonzero, METH_VARARGS | METH_KEYWORDS, NULL},
  {"numel", (PyCFunction)THPVariable_numel, METH_NOARGS, NULL},
  {"numpy", (PyCFunction)THPVariable_numpy, METH_NOARGS, NULL},
  {"record_stream", (PyCFunction)THPVariable_record_stream, METH_O, NULL},
  {"requires_grad_", (PyCFunction)(void(*)(void))THPVariable_requires_grad_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"short", (PyCFunction)(void(*)(void))THPVariable_short, METH_VARARGS | METH_KEYWORDS, NULL},
  {"size", (PyCFunction)(void(*)(void))THPVariable_size, METH_VARARGS | METH_KEYWORDS, NULL},
  {"storage", (PyCFunction)THPVariable_storage, METH_NOARGS, NULL},
  {"storage_offset", (PyCFunction)THPVariable_storage_offset, METH_NOARGS, NULL},
  {"storage_type", (PyCFunction)THPVariable_storage_type, METH_NOARGS, NULL},
  {"stride", (PyCFunction)(void(*)(void))THPVariable_stride, METH_VARARGS | METH_KEYWORDS, NULL},
  {"to", (PyCFunction)(void(*)(void))THPVariable_to, METH_VARARGS | METH_KEYWORDS, NULL},
  {"tolist", (PyCFunction)THPVariable_tolist, METH_NOARGS, NULL},
  {"type", (PyCFunction)(void(*)(void))THPVariable_type, METH_VARARGS | METH_KEYWORDS, NULL},
  ${py_method_defs}
  {NULL}
};

}} // namespace torch::autograd
