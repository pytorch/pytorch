// @generated from tools/autograd/templates/python_variable_methods.cpp

#include <Python.h>

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Size.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/autograd/utils/python_error_messages.h"
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
#include <ATen/core/EnableNamedTensor.h>

#include <ATen/ATen.h>
#include "c10/util/Optional.h"

#include "python_variable_methods_dispatch.h"

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
#ifdef BUILD_NAMEDTENSOR
    "size(Dimname dim)",
#endif
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
#ifdef BUILD_NAMEDTENSOR
  else if (r.idx == 2) {
    if (jit::tracer::isTracing()) {
      TORCH_INTERNAL_ASSERT("NYI: Named tensors w/ JIT");
    }
    return wrap(self_.size(r.dimname(0)));
  }
#endif
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_stride(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "stride(int64_t dim)",
    "stride()",
#ifdef BUILD_NAMEDTENSOR
    "stride(Dimname dim)",
#endif
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
#ifdef BUILD_NAMEDTENSOR
  else if (r.idx == 2) {
    return wrap(self_.stride(r.dimname(0)));
  }
#endif
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_get_device(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(self.get_device());
  END_HANDLE_TH_ERRORS
}

#ifdef BUILD_NAMEDTENSOR
static PyObject * THPVariable_has_names(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(self.has_names());
  END_HANDLE_TH_ERRORS
}
#endif

static PyObject * THPVariable_data_ptr(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(self.data_ptr());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_storage_offset(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(self.storage_offset());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_dim(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
   return THPUtils_packInt64(self_.dim());
   END_HANDLE_TH_ERRORS
}

static Tensor dispatch_contiguous(const Tensor & self, at::MemoryFormat memory_format) {
  AutoNoGIL no_gil;
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
  AutoNoGIL no_gil;
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
  AutoNoGIL no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.item<double>();
}

static std::complex<double> dispatch_to_CComplexDouble(const Tensor & self) {
  AutoNoGIL no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.item<std::complex<double>>();
}

static int64_t dispatch_to_CLong(const Tensor & self) {
  AutoNoGIL no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  if (self.numel() != 1) {
    throw ValueError("only one element tensors can be converted to Python scalars");
  }
  return self.item<int64_t>();
}

static bool dispatch_to_Bool(const Tensor & self) {
  AutoNoGIL no_gil;
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
  AutoNoGIL no_gil;
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

static Tensor dispatch_to(const Tensor & self, Device device, bool non_blocking, bool copy) {
  AutoNoGIL no_gil;
  // NOTE: this is where we record aten::to in the graph during tracing. However, the behavior of aten::to
  // is different with respect to TensorOptions fields that are not present: aten::to inherits fields that
  // are missing from the self argument while the tracer assumes that they should be populated with the
  // default values (eg. float for scalar type). By explicitly copying over the tensor options here we fully
  // specify all tensor options and thus record the proper trace
  return self.to(self.options().device(device), non_blocking, copy);
}

static Tensor dispatch_to(const Tensor & self, ScalarType dtype, bool non_blocking, bool copy) {
  AutoNoGIL no_gil;
  return self.to(dtype, non_blocking, copy);
}

static Tensor dispatch_to(const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy) {
  AutoNoGIL no_gil;
  return self.to(device, dtype, non_blocking, copy);
}

static PyObject * THPVariable_cpu(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
   return THPVariable_Wrap(dispatch_to(self_, at::Device(at::DeviceType::CPU), false, false));
   END_HANDLE_TH_ERRORS
}

static Tensor dispatch_nonzero(const Tensor & self) {
  AutoNoGIL no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero();
}

static std::vector<Tensor> dispatch_nonzero_numpy(const Tensor & self) {
  AutoNoGIL no_gil;
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
    "cuda(Device? device=None, bool non_blocking=False)",
    "cuda(Device? device=None, bool async=False)|deprecated"
  });
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto device = r.isNone(0) ? at::Device(at::DeviceType::CUDA) : r.device(0);
  TORCH_CHECK(device.is_cuda(), "Invalid device, must be cuda device");
  torch::utils::cuda_lazy_init();
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(1), false));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_to_type(PyObject* self, ScalarType scalarType) {
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return THPVariable_Wrap(dispatch_to(self_, scalarType, false, false));
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

static PyObject * THPVariable_bool(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::Bool);
}

static PyObject * THPVariable_bfloat16(PyObject* self, PyObject* args) {
  return THPVariable_to_type(self, ScalarType::BFloat16);
}

static PyObject * THPVariable_element_size(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  return THPUtils_packInt64(self_.element_size());
  END_HANDLE_TH_ERRORS
}

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
  void* data = self_.data_ptr();
  c10::cuda::CUDACachingAllocator::recordStream(data, at::cuda::CUDAStream::unpack(((THCPStream*)arg)->cdata));
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
  OptionalDeviceGuard device_guard(device_of(self_));
  return THPVariable_Wrap(torch::utils::legacy_tensor_new(legacyExtractTypeId(self_), self_.scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_ones(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  OptionalDeviceGuard device_guard(device_of(self_));
  return THPVariable_Wrap(torch::utils::new_ones(legacyExtractTypeId(self_), self_.scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  OptionalDeviceGuard device_guard(device_of(self_));
  return THPVariable_Wrap(torch::utils::new_tensor(legacyExtractTypeId(self_), self_.scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_zeros(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  OptionalDeviceGuard device_guard(device_of(self_));
  return THPVariable_Wrap(torch::utils::new_zeros(legacyExtractTypeId(self_), self_.scalar_type(), args, kwargs));
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
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (device && device->is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
  if (!device && !scalarType && !copy) {
    Py_INCREF(self);
    return self;
  } else if (!device) {
    return THPVariable_Wrap(dispatch_to(self_, *scalarType, non_blocking, copy));
  } else if (!scalarType) {
    return THPVariable_Wrap(dispatch_to(self_, *device, non_blocking, copy));
  } else {
    return THPVariable_Wrap(dispatch_to(self_, *device, *scalarType, non_blocking, copy));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

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
  ScalarType scalar_type;
  Device device = self_.device();
  if (is_dtype) {
    scalar_type = r.scalartype(0);
  } else {
    at::DeprecatedTypeProperties* type = torch::utils::type_from_string(type_name);
    scalar_type = type->scalarType();
    auto device_type = backendToDeviceType(type->backend());
    if (device_type != device.type()) {
      device = at::Device(device_type);
    }
  }
  if (device.is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
  return THPVariable_Wrap(dispatch_to(self_, device, scalar_type, /*non_blocking=*/ r.toBool(1), /*copy=*/ false));
  END_HANDLE_TH_ERRORS
}

// generated methods start here

static PyObject * THPVariable___and__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__and__(Tensor other)",
    "__and__(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch___and__(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___and__(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___iand__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__iand__(Tensor other)",
    "__iand__(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch___iand__(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___iand__(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___ilshift__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__ilshift__(Tensor other)",
    "__ilshift__(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch___ilshift__(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___ilshift__(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___ior__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__ior__(Tensor other)",
    "__ior__(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch___ior__(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___ior__(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___irshift__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__irshift__(Tensor other)",
    "__irshift__(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch___irshift__(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___irshift__(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___ixor__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__ixor__(Tensor other)",
    "__ixor__(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch___ixor__(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___ixor__(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___lshift__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__lshift__(Tensor other)",
    "__lshift__(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch___lshift__(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___lshift__(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___or__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__or__(Tensor other)",
    "__or__(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch___or__(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___or__(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___rshift__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__rshift__(Tensor other)",
    "__rshift__(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch___rshift__(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___rshift__(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable___xor__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__xor__(Tensor other)",
    "__xor__(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch___xor__(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch___xor__(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__coalesced_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_coalesced_(bool coalesced)",
  }, /*traceable=*/false);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch__coalesced_(self, r.toBool(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__dimI(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch__dimI(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__dimV(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch__dimV(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__indices(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch__indices(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__nnz(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch__nnz(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__values(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch__values(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_abs(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_abs(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_abs_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_abs_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_acos(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_acos(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_acos_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_acos_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "add(Scalar alpha, Tensor other)|deprecated",
    "add(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_add(self, r.scalar(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_add(self, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_add_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "add_(Scalar alpha, Tensor other)|deprecated",
    "add_(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_add_(self, r.scalar(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_add_(self, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addbmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addbmm(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_addbmm(r.scalar(0), self, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addbmm(r.scalar(0), self, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addbmm(self, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addbmm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addbmm_(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm_(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm_(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_addbmm_(r.scalar(0), self, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addbmm_(r.scalar(0), self, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addbmm_(self, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addcdiv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addcdiv(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcdiv(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_addcdiv(self, r.scalar(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addcdiv(self, r.tensor(0), r.tensor(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addcdiv_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addcdiv_(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcdiv_(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_addcdiv_(self, r.scalar(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addcdiv_(self, r.tensor(0), r.tensor(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addcmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addcmul(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcmul(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_addcmul(self, r.scalar(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addcmul(self, r.tensor(0), r.tensor(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addcmul_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addcmul_(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcmul_(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_addcmul_(self, r.scalar(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addcmul_(self, r.tensor(0), r.tensor(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmm(Scalar beta, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "addmm(Scalar beta, Tensor mat1, Tensor mat2)|deprecated",
    "addmm(Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_addmm(r.scalar(0), self, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addmm(r.scalar(0), self, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addmm(self, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addmm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmm_(Scalar beta, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "addmm_(Scalar beta, Tensor mat1, Tensor mat2)|deprecated",
    "addmm_(Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_addmm_(r.scalar(0), self, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addmm_(r.scalar(0), self, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addmm_(self, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addmv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmv(Scalar beta, Scalar alpha, Tensor mat, Tensor vec)|deprecated",
    "addmv(Scalar beta, Tensor mat, Tensor vec)|deprecated",
    "addmv(Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_addmv(r.scalar(0), self, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addmv(r.scalar(0), self, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addmv(self, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addmv_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmv_(Scalar beta, Scalar alpha, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Scalar beta, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_addmv_(r.scalar(0), self, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addmv_(r.scalar(0), self, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addmv_(self, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addr(Scalar beta, Scalar alpha, Tensor vec1, Tensor vec2)|deprecated",
    "addr(Scalar beta, Tensor vec1, Tensor vec2)|deprecated",
    "addr(Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_addr(r.scalar(0), self, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addr(r.scalar(0), self, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addr(self, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_addr_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addr_(Scalar beta, Scalar alpha, Tensor vec1, Tensor vec2)|deprecated",
    "addr_(Scalar beta, Tensor vec1, Tensor vec2)|deprecated",
    "addr_(Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_addr_(r.scalar(0), self, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_addr_(r.scalar(0), self, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_addr_(self, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_align_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "align_as(Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_align_as(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_align_to(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "align_to(DimnameList names)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_align_to(self, r.dimnamelist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_all(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "all()",
    "all(Dimname dim, bool keepdim=False)",
    "all(int64_t dim, bool keepdim=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_all(self));
  } else if (r.idx == 1) {
    return wrap(dispatch_all(self, r.dimname(0), r.toBool(1)));
  } else if (r.idx == 2) {
    return wrap(dispatch_all(self, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_allclose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "allclose(Tensor other, double rtol=1e-05, double atol=1e-08, bool equal_nan=False)",
  }, /*traceable=*/false);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_allclose(self, r.tensor(0), r.toDouble(1), r.toDouble(2), r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_any(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "any()",
    "any(Dimname dim, bool keepdim=False)",
    "any(int64_t dim, bool keepdim=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_any(self));
  } else if (r.idx == 1) {
    return wrap(dispatch_any(self, r.dimname(0), r.toBool(1)));
  } else if (r.idx == 2) {
    return wrap(dispatch_any(self, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_argmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "argmax(int64_t? dim=None, bool keepdim=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_argmax(self, r.toInt64Optional(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_argmin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "argmin(int64_t? dim=None, bool keepdim=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_argmin(self, r.toInt64Optional(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_argsort(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "argsort(Dimname dim, bool descending=False)",
    "argsort(int64_t dim=-1, bool descending=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_argsort(self, r.dimname(0), r.toBool(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_argsort(self, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_as_strided(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "as_strided(IntArrayRef size, IntArrayRef stride, int64_t? storage_offset=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_as_strided(self, r.intlist(0), r.intlist(1), r.toInt64Optional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_as_strided_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "as_strided_(IntArrayRef size, IntArrayRef stride, int64_t? storage_offset=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_as_strided_(self, r.intlist(0), r.intlist(1), r.toInt64Optional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_asin(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_asin(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_asin_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_asin_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_atan(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_atan(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_atan2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan2(Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_atan2(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_atan2_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan2_(Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_atan2_(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_atan_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_atan_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_backward(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "backward(Tensor? gradient=None, bool keep_graph=False, bool create_graph=False)",
  }, /*traceable=*/false);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    dispatch_backward(self, r.tensor(0), r.toBool(1), r.toBool(2));
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_baddbmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "baddbmm(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_baddbmm(r.scalar(0), self, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_baddbmm(r.scalar(0), self, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_baddbmm(self, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_baddbmm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "baddbmm_(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm_(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm_(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_baddbmm_(r.scalar(0), self, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_baddbmm_(r.scalar(0), self, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_baddbmm_(self, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bernoulli(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bernoulli(*, Generator generator=None)",
    "bernoulli(double p, *, Generator generator=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_bernoulli(self, r.generator(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_bernoulli(self, r.toDouble(0), r.generator(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bernoulli_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bernoulli_(Tensor p, *, Generator generator=None)",
    "bernoulli_(double p=0.5, *, Generator generator=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_bernoulli_(self, r.tensor(0), r.generator(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_bernoulli_(self, r.toDouble(0), r.generator(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bincount(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bincount(Tensor? weights=None, int64_t minlength=0)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_bincount(self, r.tensor(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bitwise_not(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_bitwise_not(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bitwise_not_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_bitwise_not_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_bmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bmm(Tensor mat2)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_bmm(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cauchy_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cauchy_(double median=0, double sigma=1, *, Generator generator=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_cauchy_(self, r.toDouble(0), r.toDouble(1), r.generator(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ceil(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_ceil(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ceil_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_ceil_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cholesky(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cholesky(bool upper=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_cholesky(self, r.toBool(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cholesky_inverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cholesky_inverse(bool upper=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_cholesky_inverse(self, r.toBool(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cholesky_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cholesky_solve(Tensor input2, bool upper=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_cholesky_solve(self, r.tensor(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_chunk(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "chunk(int64_t chunks, int64_t dim=0)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_chunk(self, r.toInt64(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clamp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp(Scalar? min=None, Scalar? max=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_clamp(self, r.scalarOptional(0), r.scalarOptional(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clamp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_(Scalar? min=None, Scalar? max=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_clamp_(self, r.scalarOptional(0), r.scalarOptional(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clamp_max(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_max(Scalar max)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_clamp_max(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clamp_max_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_max_(Scalar max)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_clamp_max_(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clamp_min(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_min(Scalar min)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_clamp_min(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clamp_min_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp_min_(Scalar min)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_clamp_min_(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_clone(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_clone(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_coalesce(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_coalesce(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cos(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_cos(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cos_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_cos_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cosh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_cosh(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cosh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_cosh_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cross(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cross(Tensor other, int64_t? dim=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_cross(self, r.tensor(0), r.toInt64Optional(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cumprod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cumprod(Dimname dim, *, ScalarType? dtype=None)",
    "cumprod(int64_t dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_cumprod(self, r.dimname(0), r.scalartypeOptional(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_cumprod(self, r.toInt64(0), r.scalartypeOptional(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_cumsum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cumsum(Dimname dim, *, ScalarType? dtype=None)",
    "cumsum(int64_t dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_cumsum(self, r.dimname(0), r.scalartypeOptional(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_cumsum(self, r.toInt64(0), r.scalartypeOptional(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_dense_dim(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_dense_dim(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_dequantize(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_dequantize(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_det(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_det(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_detach(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_detach(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_detach_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_detach_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_diag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diag(int64_t diagonal=0)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_diag(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_diag_embed(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diag_embed(int64_t offset=0, int64_t dim1=-2, int64_t dim2=-1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_diag_embed(self, r.toInt64(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_diagflat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diagflat(int64_t offset=0)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_diagflat(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_diagonal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diagonal(int64_t offset=0, int64_t dim1=0, int64_t dim2=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_diagonal(self, r.toInt64(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_digamma(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_digamma(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_digamma_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_digamma_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_dist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dist(Tensor other, Scalar p=2)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_dist(self, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_div(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "div(Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_div(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_div_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "div_(Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_div_(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_dot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dot(Tensor tensor)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_dot(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_eig(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eig(bool eigenvectors=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  static PyStructSequence_Field fields0[] = {
    {"eigenvalues", ""}, {"eigenvectors", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.eig", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  if (r.idx == 0) {
    return wrap(&type0, dispatch_eig(self, r.toBool(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_eq(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eq(Tensor other)",
    "eq(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_eq(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_eq(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_eq_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eq_(Tensor other)",
    "eq_(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_eq_(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_eq_(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_equal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "equal(Tensor other)",
  }, /*traceable=*/false);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_equal(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erf(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_erf(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erf_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_erf_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erfc(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_erfc(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erfc_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_erfc_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erfinv(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_erfinv(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_erfinv_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_erfinv_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_exp(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_exp(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_exp_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_exp_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_expand(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "expand(IntArrayRef size, *, bool implicit=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_expand(self, r.intlist(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_expand_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "expand_as(Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_expand_as(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_expm1(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_expm1(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_expm1_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_expm1_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_exponential_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "exponential_(double lambd=1, *, Generator generator=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_exponential_(self, r.toDouble(0), r.generator(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_fft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft(int64_t signal_ndim, bool normalized=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_fft(self, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_fill_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fill_(Tensor value)",
    "fill_(Scalar value)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_fill_(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_fill_(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_fill_diagonal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fill_diagonal_(Scalar fill_value, bool wrap=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_fill_diagonal_(self, r.scalar(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_flatten(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "flatten(Dimname start_dim, Dimname end_dim, Dimname out_dim)",
    "flatten(DimnameList dims, Dimname out_dim)",
    "flatten(int64_t start_dim, int64_t end_dim, Dimname out_dim)",
    "flatten(int64_t start_dim=0, int64_t end_dim=-1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_flatten(self, r.dimname(0), r.dimname(1), r.dimname(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_flatten(self, r.dimnamelist(0), r.dimname(1)));
  } else if (r.idx == 2) {
    return wrap(dispatch_flatten(self, r.toInt64(0), r.toInt64(1), r.dimname(2)));
  } else if (r.idx == 3) {
    return wrap(dispatch_flatten(self, r.toInt64(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_flip(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "flip(IntArrayRef dims)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_flip(self, r.intlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_floor(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_floor(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_floor_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_floor_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_fmod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fmod(Tensor other)",
    "fmod(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_fmod(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_fmod(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_fmod_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fmod_(Tensor other)",
    "fmod_(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_fmod_(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_fmod_(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_frac(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_frac(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_frac_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_frac_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gather(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gather(Dimname dim, Tensor index, *, bool sparse_grad=False)",
    "gather(int64_t dim, Tensor index, *, bool sparse_grad=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_gather(self, r.dimname(0), r.tensor(1), r.toBool(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_gather(self, r.toInt64(0), r.tensor(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ge(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ge(Tensor other)",
    "ge(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_ge(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_ge(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ge_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ge_(Tensor other)",
    "ge_(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_ge_(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_ge_(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_geometric_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "geometric_(double p, *, Generator generator=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_geometric_(self, r.toDouble(0), r.generator(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_geqrf(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field fields0[] = {
    {"a", ""}, {"tau", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.geqrf", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(&type0, dispatch_geqrf(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ger(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ger(Tensor vec2)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_ger(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gt(Tensor other)",
    "gt(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_gt(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_gt(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_gt_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gt_(Tensor other)",
    "gt_(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_gt_(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_gt_(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_hardshrink(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardshrink(Scalar lambd=0.5)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_hardshrink(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_histc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "histc(int64_t bins=100, Scalar min=0, Scalar max=0)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_histc(self, r.toInt64(0), r.scalar(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ifft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ifft(int64_t signal_ndim, bool normalized=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_ifft(self, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_add(Dimname dim, Tensor index, Tensor source)",
    "index_add(int64_t dim, Tensor index, Tensor source)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_index_add(self, r.dimname(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_index_add(self, r.toInt64(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_add_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_add_(int64_t dim, Tensor index, Tensor source)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_index_add_(self, r.toInt64(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_copy(Dimname dim, Tensor index, Tensor source)",
    "index_copy(int64_t dim, Tensor index, Tensor source)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_index_copy(self, r.dimname(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_index_copy(self, r.toInt64(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_copy_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_copy_(Dimname dim, Tensor index, Tensor source)",
    "index_copy_(int64_t dim, Tensor index, Tensor source)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_index_copy_(self, r.dimname(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_index_copy_(self, r.toInt64(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_fill(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_fill(Dimname dim, Tensor index, Tensor value)",
    "index_fill(int64_t dim, Tensor index, Tensor value)",
    "index_fill(Dimname dim, Tensor index, Scalar value)",
    "index_fill(int64_t dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_index_fill(self, r.dimname(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_index_fill(self, r.toInt64(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_index_fill(self, r.dimname(0), r.tensor(1), r.scalar(2)));
  } else if (r.idx == 3) {
    return wrap(dispatch_index_fill(self, r.toInt64(0), r.tensor(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_fill_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_fill_(int64_t dim, Tensor index, Tensor value)",
    "index_fill_(int64_t dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_index_fill_(self, r.toInt64(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_index_fill_(self, r.toInt64(0), r.tensor(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_put(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_put(TensorList? indices, Tensor values, bool accumulate=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_index_put(self, r.tensorlist(0), r.tensor(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_put_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_put_(TensorList? indices, Tensor values, bool accumulate=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_index_put_(self, r.tensorlist(0), r.tensor(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_index_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_select(Dimname dim, Tensor index)",
    "index_select(int64_t dim, Tensor index)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_index_select(self, r.dimname(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_index_select(self, r.toInt64(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_indices(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_indices(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_int_repr(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_int_repr(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_inverse(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_inverse(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_irfft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "irfft(int64_t signal_ndim, bool normalized=False, bool onesided=True, IntArrayRef signal_sizes=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_irfft(self, r.toInt64(0), r.toBool(1), r.toBool(2), r.intlist(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_coalesced(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_is_coalesced(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_complex(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_is_complex(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_distributed(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_is_distributed(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_floating_point(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_is_floating_point(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_nonzero(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_is_nonzero(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_pinned(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_is_pinned(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_same_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_same_size(Tensor other)",
  }, /*traceable=*/false);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_is_same_size(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_set_to(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_set_to(Tensor tensor)",
  }, /*traceable=*/false);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_is_set_to(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_is_signed(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_is_signed(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_isclose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isclose(Tensor other, double rtol=1e-05, double atol=1e-08, bool equal_nan=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_isclose(self, r.tensor(0), r.toDouble(1), r.toDouble(2), r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_kthvalue(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "kthvalue(int64_t k, Dimname dim, bool keepdim=False)",
    "kthvalue(int64_t k, int64_t dim=-1, bool keepdim=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  static PyStructSequence_Field fields0[] = {
    {"values", ""}, {"indices", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.kthvalue", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  static PyStructSequence_Field fields1[] = {
    {"values", ""}, {"indices", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc1 = {
    "torch.return_types.kthvalue", nullptr,
    fields1, 2
  };
  static PyTypeObject type1;
  static bool namedtuple_type_initialized1 = false;
  if (!namedtuple_type_initialized1) {
    PyStructSequence_InitType(&type1, &desc1);
    type1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized1 = true;
  }
  if (r.idx == 0) {
    return wrap(&type1, dispatch_kthvalue(self, r.toInt64(0), r.dimname(1), r.toBool(2)));
  } else if (r.idx == 1) {
    return wrap(&type0, dispatch_kthvalue(self, r.toInt64(0), r.toInt64(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_le(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "le(Tensor other)",
    "le(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_le(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_le(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_le_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "le_(Tensor other)",
    "le_(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_le_(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_le_(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lerp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lerp(Tensor end, Tensor weight)",
    "lerp(Tensor end, Scalar weight)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_lerp(self, r.tensor(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_lerp(self, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lerp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lerp_(Tensor end, Tensor weight)",
    "lerp_(Tensor end, Scalar weight)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_lerp_(self, r.tensor(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_lerp_(self, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lgamma(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_lgamma(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lgamma_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_lgamma_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_log(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log10(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_log10(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log10_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_log10_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log1p(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_log1p(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log1p_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_log1p_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log2(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_log2(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log2_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_log2_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_log_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log_normal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log_normal_(double mean=1, double std=2, *, Generator generator=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_log_normal_(self, r.toDouble(0), r.toDouble(1), r.generator(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log_softmax(Dimname dim, *, ScalarType? dtype=None)",
    "log_softmax(int64_t dim, ScalarType? dtype=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_log_softmax(self, r.dimname(0), r.scalartypeOptional(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_log_softmax(self, r.toInt64(0), r.scalartypeOptional(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_logdet(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_logdet(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_logical_not(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_logical_not(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_logical_not_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_logical_not_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_logical_xor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logical_xor(Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_logical_xor(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_logical_xor_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logical_xor_(Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_logical_xor_(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_logsumexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logsumexp(DimnameList[1] dim, bool keepdim=False)",
    "logsumexp(IntArrayRef[1] dim, bool keepdim=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_logsumexp(self, r.dimnamelist(0), r.toBool(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_logsumexp(self, r.intlist(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lstsq(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lstsq(Tensor A)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  static PyStructSequence_Field fields0[] = {
    {"solution", ""}, {"QR", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.lstsq", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  if (r.idx == 0) {
    return wrap(&type0, dispatch_lstsq(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lt(Tensor other)",
    "lt(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_lt(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_lt(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lt_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lt_(Tensor other)",
    "lt_(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_lt_(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_lt_(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_lu_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lu_solve(Tensor LU_data, Tensor LU_pivots)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_lu_solve(self, r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_masked_fill(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_fill(Tensor mask, Tensor value)",
    "masked_fill(Tensor mask, Scalar value)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_masked_fill(self, r.tensor(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_masked_fill(self, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_masked_fill_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_fill_(Tensor mask, Tensor value)",
    "masked_fill_(Tensor mask, Scalar value)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_masked_fill_(self, r.tensor(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_masked_fill_(self, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_masked_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_scatter(Tensor mask, Tensor source)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_masked_scatter(self, r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_masked_scatter_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_scatter_(Tensor mask, Tensor source)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_masked_scatter_(self, r.tensor(0), r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_masked_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_select(Tensor mask)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_masked_select(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_matmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "matmul(Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_matmul(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_matrix_power(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "matrix_power(int64_t n)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_matrix_power(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_max(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max()",
    "max(Dimname dim, bool keepdim=False)",
    "max(Tensor other)",
    "max(int64_t dim, bool keepdim=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  static PyStructSequence_Field fields0[] = {
    {"values", ""}, {"indices", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.max", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  static PyStructSequence_Field fields1[] = {
    {"values", ""}, {"indices", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc1 = {
    "torch.return_types.max", nullptr,
    fields1, 2
  };
  static PyTypeObject type1;
  static bool namedtuple_type_initialized1 = false;
  if (!namedtuple_type_initialized1) {
    PyStructSequence_InitType(&type1, &desc1);
    type1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized1 = true;
  }
  if (r.idx == 0) {
    return wrap(dispatch_max(self));
  } else if (r.idx == 1) {
    return wrap(&type1, dispatch_max(self, r.dimname(0), r.toBool(1)));
  } else if (r.idx == 2) {
    return wrap(dispatch_max(self, r.tensor(0)));
  } else if (r.idx == 3) {
    return wrap(&type0, dispatch_max(self, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mean(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mean(*, ScalarType? dtype=None)",
    "mean(DimnameList[1] dim, bool keepdim=False, *, ScalarType? dtype=None)",
    "mean(IntArrayRef[1] dim, bool keepdim=False, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_mean(self, r.scalartypeOptional(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_mean(self, r.dimnamelist(0), r.toBool(1), r.scalartypeOptional(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_mean(self, r.intlist(0), r.toBool(1), r.scalartypeOptional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_median(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "median()",
    "median(Dimname dim, bool keepdim=False)",
    "median(int64_t dim, bool keepdim=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  static PyStructSequence_Field fields0[] = {
    {"values", ""}, {"indices", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.median", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  static PyStructSequence_Field fields1[] = {
    {"values", ""}, {"indices", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc1 = {
    "torch.return_types.median", nullptr,
    fields1, 2
  };
  static PyTypeObject type1;
  static bool namedtuple_type_initialized1 = false;
  if (!namedtuple_type_initialized1) {
    PyStructSequence_InitType(&type1, &desc1);
    type1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized1 = true;
  }
  if (r.idx == 0) {
    return wrap(dispatch_median(self));
  } else if (r.idx == 1) {
    return wrap(&type1, dispatch_median(self, r.dimname(0), r.toBool(1)));
  } else if (r.idx == 2) {
    return wrap(&type0, dispatch_median(self, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_min(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "min()",
    "min(Dimname dim, bool keepdim=False)",
    "min(Tensor other)",
    "min(int64_t dim, bool keepdim=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  static PyStructSequence_Field fields0[] = {
    {"values", ""}, {"indices", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.min", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  static PyStructSequence_Field fields1[] = {
    {"values", ""}, {"indices", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc1 = {
    "torch.return_types.min", nullptr,
    fields1, 2
  };
  static PyTypeObject type1;
  static bool namedtuple_type_initialized1 = false;
  if (!namedtuple_type_initialized1) {
    PyStructSequence_InitType(&type1, &desc1);
    type1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized1 = true;
  }
  if (r.idx == 0) {
    return wrap(dispatch_min(self));
  } else if (r.idx == 1) {
    return wrap(&type1, dispatch_min(self, r.dimname(0), r.toBool(1)));
  } else if (r.idx == 2) {
    return wrap(dispatch_min(self, r.tensor(0)));
  } else if (r.idx == 3) {
    return wrap(&type0, dispatch_min(self, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mm(Tensor mat2)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_mm(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mode(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mode(Dimname dim, bool keepdim=False)",
    "mode(int64_t dim=-1, bool keepdim=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  static PyStructSequence_Field fields0[] = {
    {"values", ""}, {"indices", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.mode", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  static PyStructSequence_Field fields1[] = {
    {"values", ""}, {"indices", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc1 = {
    "torch.return_types.mode", nullptr,
    fields1, 2
  };
  static PyTypeObject type1;
  static bool namedtuple_type_initialized1 = false;
  if (!namedtuple_type_initialized1) {
    PyStructSequence_InitType(&type1, &desc1);
    type1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized1 = true;
  }
  if (r.idx == 0) {
    return wrap(&type1, dispatch_mode(self, r.dimname(0), r.toBool(1)));
  } else if (r.idx == 1) {
    return wrap(&type0, dispatch_mode(self, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mul(Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_mul(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mul_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mul_(Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_mul_(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_multinomial(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "multinomial(int64_t num_samples, bool replacement=False, *, Generator generator=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_multinomial(self, r.toInt64(0), r.toBool(1), r.generator(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mv(Tensor vec)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_mv(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mvlgamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mvlgamma(int64_t p)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_mvlgamma(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mvlgamma_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mvlgamma_(int64_t p)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_mvlgamma_(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_narrow(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "narrow(int64_t dim, int64_t start, int64_t length)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_narrow(self, r.toInt64(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_narrow_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "narrow_copy(int64_t dim, int64_t start, int64_t length)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_narrow_copy(self, r.toInt64(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ne(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ne(Tensor other)",
    "ne(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_ne(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_ne(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ne_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ne_(Tensor other)",
    "ne_(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_ne_(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_ne_(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_neg(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_neg(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_neg_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_neg_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_new_empty(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "new_empty(IntArrayRef size, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    auto size = r.intlist(0);
    auto dtype = r.scalartypeWithDefault(1, self.scalar_type());
    auto device = r.deviceWithDefault(3, self.device());
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layoutWithDefault(2, *torch::getLayout(self.type().backend())).layout)
        .requires_grad(r.toBool(5))
        .pinned_memory(r.toBool(4));
    return wrap(dispatch_new_empty(self, size, options));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_new_full(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "new_full(IntArrayRef size, Scalar fill_value, *, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    auto size = r.intlist(0);
    auto fill_value = r.scalar(1);
    auto dtype = r.scalartypeWithDefault(2, self.scalar_type());
    auto device = r.deviceWithDefault(4, self.device());
    const auto options = TensorOptions()
        .dtype(dtype)
        .device(device)
        .layout(r.layoutWithDefault(3, *torch::getLayout(self.type().backend())).layout)
        .requires_grad(r.toBool(6))
        .pinned_memory(r.toBool(5));
    return wrap(dispatch_new_full(self, size, fill_value, options));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "norm(Scalar p=2)",
    "norm(Scalar? p, *, ScalarType dtype)",
    "norm(Scalar? p, DimnameList[1] dim, bool keepdim, *, ScalarType dtype)",
    "norm(Scalar? p, DimnameList[1] dim, bool keepdim=False)",
    "norm(Scalar? p, IntArrayRef[1] dim, bool keepdim, *, ScalarType dtype)",
    "norm(Scalar? p, IntArrayRef[1] dim, bool keepdim=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_norm(self, r.scalar(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_norm(self, r.scalarOptional(0), r.scalartype(1)));
  } else if (r.idx == 2) {
    return wrap(dispatch_norm(self, r.scalarOptional(0), r.dimnamelist(1), r.toBool(2), r.scalartype(3)));
  } else if (r.idx == 3) {
    return wrap(dispatch_norm(self, r.scalarOptional(0), r.dimnamelist(1), r.toBool(2)));
  } else if (r.idx == 4) {
    return wrap(dispatch_norm(self, r.scalarOptional(0), r.intlist(1), r.toBool(2), r.scalartype(3)));
  } else if (r.idx == 5) {
    return wrap(dispatch_norm(self, r.scalarOptional(0), r.intlist(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_normal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "normal_(double mean=0, double std=1, *, Generator generator=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_normal_(self, r.toDouble(0), r.toDouble(1), r.generator(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_numel(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_numel(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_orgqr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "orgqr(Tensor input2)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_orgqr(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_ormqr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ormqr(Tensor input2, Tensor input3, bool left=True, bool transpose=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_ormqr(self, r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_permute(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "permute(IntArrayRef dims)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_permute(self, r.intlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pin_memory(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_pin_memory(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pinverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pinverse(double rcond=1e-15)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_pinverse(self, r.toDouble(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_polygamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "polygamma(int64_t n, )",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_polygamma(r.toInt64(0), self));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_polygamma_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "polygamma_(int64_t n)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_polygamma_(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pow(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pow(Tensor exponent)",
    "pow(Scalar exponent)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_pow(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_pow(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_pow_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pow_(Tensor exponent)",
    "pow_(Scalar exponent)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_pow_(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_pow_(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_prelu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "prelu(Tensor weight)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_prelu(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_prod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "prod(*, ScalarType? dtype=None)",
    "prod(Dimname dim, bool keepdim=False, *, ScalarType? dtype=None)",
    "prod(int64_t dim, bool keepdim=False, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_prod(self, r.scalartypeOptional(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_prod(self, r.dimname(0), r.toBool(1), r.scalartypeOptional(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_prod(self, r.toInt64(0), r.toBool(1), r.scalartypeOptional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_put_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "put_(Tensor index, Tensor source, bool accumulate=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_put_(self, r.tensor(0), r.tensor(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_q_per_channel_axis(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_q_per_channel_axis(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_q_per_channel_scales(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_q_per_channel_scales(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_q_per_channel_zero_points(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_q_per_channel_zero_points(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_q_scale(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_q_scale(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_q_zero_point(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_q_zero_point(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_qr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "qr(bool some=True)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  static PyStructSequence_Field fields0[] = {
    {"Q", ""}, {"R", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.qr", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  if (r.idx == 0) {
    return wrap(&type0, dispatch_qr(self, r.toBool(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_qscheme(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_qscheme(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_random_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "random_(*, Generator generator=None)",
    "random_(int64_t from, int64_t to, *, Generator generator=None)",
    "random_(int64_t to, *, Generator generator=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_random_(self, r.generator(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_random_(self, r.toInt64(0), r.toInt64(1), r.generator(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_random_(self, r.toInt64(0), r.generator(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_reciprocal(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_reciprocal(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_reciprocal_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_reciprocal_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_refine_names(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "refine_names(DimnameList names)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_refine_names(self, r.dimnamelist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_relu(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_relu(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_relu_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_relu_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_remainder(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "remainder(Tensor other)",
    "remainder(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_remainder(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_remainder(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_remainder_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "remainder_(Tensor other)",
    "remainder_(Scalar other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_remainder_(self, r.tensor(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_remainder_(self, r.scalar(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rename(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rename(DimnameList? names)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    auto __names = r.toDimnameListOptional(0);
    c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
    return wrap(dispatch_rename(self, names));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rename_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rename_(DimnameList? names)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    auto __names = r.toDimnameListOptional(0);
    c10::optional<DimnameList> names = __names ? c10::make_optional(DimnameList(__names.value())) : c10::nullopt;
    return wrap(dispatch_rename_(self, names));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_renorm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "renorm(Scalar p, int64_t dim, Scalar maxnorm)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_renorm(self, r.scalar(0), r.toInt64(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_renorm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "renorm_(Scalar p, int64_t dim, Scalar maxnorm)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_renorm_(self, r.scalar(0), r.toInt64(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_repeat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "repeat(IntArrayRef repeats)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_repeat(self, r.intlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_repeat_interleave(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "repeat_interleave(Tensor repeats, int64_t? dim=None)",
    "repeat_interleave(int64_t repeats, int64_t? dim=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_repeat_interleave(self, r.tensor(0), r.toInt64Optional(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_repeat_interleave(self, r.toInt64(0), r.toInt64Optional(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_reshape(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reshape(IntArrayRef shape)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_reshape(self, r.intlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_reshape_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reshape_as(Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_reshape_as(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_resize_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "resize_(IntArrayRef size)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_resize_(self, r.intlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_resize_as_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "resize_as_(Tensor the_template)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_resize_as_(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rfft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rfft(int64_t signal_ndim, bool normalized=False, bool onesided=True)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_rfft(self, r.toInt64(0), r.toBool(1), r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_roll(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "roll(IntArrayRef[1] shifts, IntArrayRef[1] dims=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_roll(self, r.intlist(0), r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rot90(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rot90(int64_t k=1, IntArrayRef dims={0,1})",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_rot90(self, r.toInt64(0), r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_round(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_round(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_round_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_round_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rsqrt(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_rsqrt(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rsqrt_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_rsqrt_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "scatter(Dimname dim, Tensor index, Tensor src)",
    "scatter(int64_t dim, Tensor index, Tensor src)",
    "scatter(Dimname dim, Tensor index, Scalar value)",
    "scatter(int64_t dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_scatter(self, r.dimname(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_scatter(self, r.toInt64(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_scatter(self, r.dimname(0), r.tensor(1), r.scalar(2)));
  } else if (r.idx == 3) {
    return wrap(dispatch_scatter(self, r.toInt64(0), r.tensor(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_scatter_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "scatter_(int64_t dim, Tensor index, Tensor src)",
    "scatter_(int64_t dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_scatter_(self, r.toInt64(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_scatter_(self, r.toInt64(0), r.tensor(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_scatter_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "scatter_add(Dimname dim, Tensor index, Tensor src)",
    "scatter_add(int64_t dim, Tensor index, Tensor src)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_scatter_add(self, r.dimname(0), r.tensor(1), r.tensor(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_scatter_add(self, r.toInt64(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_scatter_add_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "scatter_add_(int64_t dim, Tensor index, Tensor src)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_scatter_add_(self, r.toInt64(0), r.tensor(1), r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "select(Dimname dim, int64_t index)",
    "select(int64_t dim, int64_t index)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_select(self, r.dimname(0), r.toInt64(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_select(self, r.toInt64(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_set_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "set_()",
    "set_(Storage source)",
    "set_(Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride=None)",
    "set_(Tensor source)",
  }, /*traceable=*/false);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_set_(self));
  } else if (r.idx == 1) {
    return wrap(dispatch_set_(self, r.storage(0)));
  } else if (r.idx == 2) {
    return wrap(dispatch_set_(self, r.storage(0), r.toInt64(1), r.intlist(2), r.intlist(3)));
  } else if (r.idx == 3) {
    return wrap(dispatch_set_(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sigmoid(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_sigmoid(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sigmoid_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_sigmoid_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sign(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_sign(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sign_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_sign_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sin(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_sin(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sin_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_sin_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sinh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_sinh(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sinh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_sinh_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_slogdet(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field fields0[] = {
    {"sign", ""}, {"logabsdet", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.slogdet", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(&type0, dispatch_slogdet(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_smm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "smm(Tensor mat2)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_smm(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "softmax(Dimname dim, *, ScalarType? dtype=None)",
    "softmax(int64_t dim, ScalarType? dtype=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_softmax(self, r.dimname(0), r.scalartypeOptional(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_softmax(self, r.toInt64(0), r.scalartypeOptional(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "solve(Tensor A)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  static PyStructSequence_Field fields0[] = {
    {"solution", ""}, {"LU", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.solve", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  if (r.idx == 0) {
    return wrap(&type0, dispatch_solve(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sort(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sort(Dimname dim, bool descending=False)",
    "sort(int64_t dim=-1, bool descending=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  static PyStructSequence_Field fields0[] = {
    {"values", ""}, {"indices", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.sort", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  static PyStructSequence_Field fields1[] = {
    {"values", ""}, {"indices", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc1 = {
    "torch.return_types.sort", nullptr,
    fields1, 2
  };
  static PyTypeObject type1;
  static bool namedtuple_type_initialized1 = false;
  if (!namedtuple_type_initialized1) {
    PyStructSequence_InitType(&type1, &desc1);
    type1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized1 = true;
  }
  if (r.idx == 0) {
    return wrap(&type1, dispatch_sort(self, r.dimname(0), r.toBool(1)));
  } else if (r.idx == 1) {
    return wrap(&type0, dispatch_sort(self, r.toInt64(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sparse_dim(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_sparse_dim(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sparse_mask(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sparse_mask(Tensor mask)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_sparse_mask(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sparse_resize_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sparse_resize_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_sparse_resize_(self, r.intlist(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sparse_resize_and_clear_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sparse_resize_and_clear_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_sparse_resize_and_clear_(self, r.intlist(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_split(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "split(int64_t split_size, int64_t dim=0)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_split(self, r.toInt64(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_split_with_sizes(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "split_with_sizes(IntArrayRef split_sizes, int64_t dim=0)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_split_with_sizes(self, r.intlist(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sqrt(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_sqrt(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sqrt_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_sqrt_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_squeeze(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "squeeze()",
    "squeeze(Dimname dim)",
    "squeeze(int64_t dim)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_squeeze(self));
  } else if (r.idx == 1) {
    return wrap(dispatch_squeeze(self, r.dimname(0)));
  } else if (r.idx == 2) {
    return wrap(dispatch_squeeze(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_squeeze_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "squeeze_()",
    "squeeze_(Dimname dim)",
    "squeeze_(int64_t dim)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_squeeze_(self));
  } else if (r.idx == 1) {
    return wrap(dispatch_squeeze_(self, r.dimname(0)));
  } else if (r.idx == 2) {
    return wrap(dispatch_squeeze_(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sspaddmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sspaddmm(Scalar beta, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Scalar beta, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_sspaddmm(r.scalar(0), self, r.scalar(1), r.tensor(2), r.tensor(3)));
  } else if (r.idx == 1) {
    return wrap(dispatch_sspaddmm(r.scalar(0), self, r.tensor(1), r.tensor(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_sspaddmm(self, r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_std(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "std(DimnameList[1] dim, bool unbiased=True, bool keepdim=False)",
    "std(IntArrayRef[1] dim, bool unbiased=True, bool keepdim=False)",
    "std(bool unbiased=True)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_std(self, r.dimnamelist(0), r.toBool(1), r.toBool(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_std(self, r.intlist(0), r.toBool(1), r.toBool(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_std(self, r.toBool(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_stft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "stft(int64_t n_fft, int64_t? hop_length=None, int64_t? win_length=None, Tensor? window=None, bool normalized=False, bool onesided=True)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_stft(self, r.toInt64(0), r.toInt64Optional(1), r.toInt64Optional(2), r.tensor(3), r.toBool(4), r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sub(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sub(Scalar alpha, Tensor other)|deprecated",
    "sub(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_sub(self, r.scalar(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_sub(self, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sub_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sub_(Scalar alpha, Tensor other)|deprecated",
    "sub_(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_sub_(self, r.scalar(0), r.tensor(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_sub_(self, r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sum(*, ScalarType? dtype=None)",
    "sum(DimnameList[1] dim, bool keepdim=False, *, ScalarType? dtype=None)",
    "sum(IntArrayRef[1] dim, bool keepdim=False, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_sum(self, r.scalartypeOptional(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_sum(self, r.dimnamelist(0), r.toBool(1), r.scalartypeOptional(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_sum(self, r.intlist(0), r.toBool(1), r.scalartypeOptional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_sum_to_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sum_to_size(IntArrayRef size)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_sum_to_size(self, r.intlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_svd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "svd(bool some=True, bool compute_uv=True)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  static PyStructSequence_Field fields0[] = {
    {"U", ""}, {"S", ""}, {"V", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.svd", nullptr,
    fields0, 3
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  if (r.idx == 0) {
    return wrap(&type0, dispatch_svd(self, r.toBool(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_symeig(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "symeig(bool eigenvectors=False, bool upper=True)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  static PyStructSequence_Field fields0[] = {
    {"eigenvalues", ""}, {"eigenvectors", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.symeig", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  if (r.idx == 0) {
    return wrap(&type0, dispatch_symeig(self, r.toBool(0), r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_t(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_t(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_t_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_t_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_take(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "take(Tensor index)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_take(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tan(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_tan(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tan_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_tan_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tanh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_tanh(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tanh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_tanh_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_to_dense(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_to_dense(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_to_mkldnn(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_to_mkldnn(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_to_sparse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "to_sparse()",
    "to_sparse(int64_t sparse_dim)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_to_sparse(self));
  } else if (r.idx == 1) {
    return wrap(dispatch_to_sparse(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_topk(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "topk(int64_t k, int64_t dim=-1, bool largest=True, bool sorted=True)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  static PyStructSequence_Field fields0[] = {
    {"values", ""}, {"indices", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.topk", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  if (r.idx == 0) {
    return wrap(&type0, dispatch_topk(self, r.toInt64(0), r.toInt64(1), r.toBool(2), r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trace(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_trace(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "transpose(Dimname dim0, Dimname dim1)",
    "transpose(int64_t dim0, int64_t dim1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_transpose(self, r.dimname(0), r.dimname(1)));
  } else if (r.idx == 1) {
    return wrap(dispatch_transpose(self, r.toInt64(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_transpose_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "transpose_(int64_t dim0, int64_t dim1)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_transpose_(self, r.toInt64(0), r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_triangular_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triangular_solve(Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  static PyStructSequence_Field fields0[] = {
    {"solution", ""}, {"cloned_coefficient", ""}, {nullptr}
  };
  static PyStructSequence_Desc desc0 = {
    "torch.return_types.triangular_solve", nullptr,
    fields0, 2
  };
  static PyTypeObject type0;
  static bool namedtuple_type_initialized0 = false;
  if (!namedtuple_type_initialized0) {
    PyStructSequence_InitType(&type0, &desc0);
    type0.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
    namedtuple_type_initialized0 = true;
  }
  if (r.idx == 0) {
    return wrap(&type0, dispatch_triangular_solve(self, r.tensor(0), r.toBool(1), r.toBool(2), r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tril(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tril(int64_t diagonal=0)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_tril(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_tril_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tril_(int64_t diagonal=0)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_tril_(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_triu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triu(int64_t diagonal=0)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_triu(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_triu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triu_(int64_t diagonal=0)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_triu_(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trunc(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_trunc(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_trunc_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_trunc_(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_type_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "type_as(Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_type_as(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_unbind(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unbind(Dimname dim)",
    "unbind(int64_t dim=0)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_unbind(self, r.dimname(0)));
  } else if (r.idx == 1) {
    return wrap(dispatch_unbind(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_unflatten(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unflatten(Dimname dim, IntArrayRef sizes, DimnameList names)",
    "unflatten(int64_t dim, IntArrayRef sizes, DimnameList names)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_unflatten(self, r.dimname(0), r.intlist(1), r.dimnamelist(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_unflatten(self, r.toInt64(0), r.intlist(1), r.dimnamelist(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_unfold(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unfold(int64_t dimension, int64_t size, int64_t step)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_unfold(self, r.toInt64(0), r.toInt64(1), r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_uniform_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "uniform_(double from=0, double to=1, *, Generator generator=None)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_uniform_(self, r.toDouble(0), r.toDouble(1), r.generator(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_unsqueeze(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unsqueeze(int64_t dim)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_unsqueeze(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_unsqueeze_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unsqueeze_(int64_t dim)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_unsqueeze_(self, r.toInt64(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_values(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_values(self));
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_var(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "var(DimnameList[1] dim, bool unbiased=True, bool keepdim=False)",
    "var(IntArrayRef[1] dim, bool unbiased=True, bool keepdim=False)",
    "var(bool unbiased=True)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_var(self, r.dimnamelist(0), r.toBool(1), r.toBool(2)));
  } else if (r.idx == 1) {
    return wrap(dispatch_var(self, r.intlist(0), r.toBool(1), r.toBool(2)));
  } else if (r.idx == 2) {
    return wrap(dispatch_var(self, r.toBool(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_view(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "view(IntArrayRef size)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_view(self, r.intlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_view_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "view_as(Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_view_as(self, r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_where(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "where(Tensor condition, Tensor other)",
  }, /*traceable=*/true);
  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (r.idx == 0) {
    return wrap(dispatch_where(r.tensor(0), self, r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_zero_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
  return wrap(dispatch_zero_(self));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_bool_scalar(PyObject* self, PyObject* args) {
  jit::tracer::warn("Converting a tensor to a Python boolean", jit::tracer::WARN_PYTHON_DATAFLOW);
  return THPVariable_is_nonzero(self, args);
}

PyMethodDef variable_methods[] = {
  {"__add__", (PyCFunction)(void(*)(void))THPVariable_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__radd__", (PyCFunction)(void(*)(void))THPVariable_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__iadd__", (PyCFunction)(void(*)(void))THPVariable_add_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__rmul__", (PyCFunction)(void(*)(void))THPVariable_mul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__mul__", (PyCFunction)(void(*)(void))THPVariable_mul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__imul__", (PyCFunction)(void(*)(void))THPVariable_mul_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__sub__", (PyCFunction)(void(*)(void))THPVariable_sub, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__isub__", (PyCFunction)(void(*)(void))THPVariable_sub_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__div__", (PyCFunction)(void(*)(void))THPVariable_div, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__truediv__", (PyCFunction)(void(*)(void))THPVariable_div, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__idiv__", (PyCFunction)(void(*)(void))THPVariable_div_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__mod__", (PyCFunction)(void(*)(void))THPVariable_remainder, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__bool__", (PyCFunction)THPVariable_bool_scalar, METH_NOARGS, NULL},
  {"__float__", (PyCFunction)THPVariable_float_scalar, METH_NOARGS, NULL},
  {"__int__", (PyCFunction)THPVariable_integral_scalar, METH_NOARGS, NULL},
  {"__long__", (PyCFunction)THPVariable_integral_scalar, METH_NOARGS, NULL},
  {"__index__", (PyCFunction)THPVariable_index_scalar, METH_NOARGS, NULL},
  {"__nonzero__", (PyCFunction)THPVariable_bool_scalar, METH_NOARGS, NULL},
  {"__invert__", (PyCFunction)THPVariable_invert, METH_NOARGS, NULL},
  {"__matmul__", (PyCFunction)(void(*)(void))THPVariable_matmul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_is_view", (PyCFunction)THPVariable__is_view, METH_NOARGS, NULL},
  {"apply_", (PyCFunction)THPVariable_apply_, METH_O, NULL},
  {"bfloat16", (PyCFunction)THPVariable_bfloat16, METH_NOARGS, NULL},
  {"byte", (PyCFunction)THPVariable_byte, METH_NOARGS, NULL},
  {"char", (PyCFunction)THPVariable_char, METH_NOARGS, NULL},
  {"contiguous", (PyCFunction)(void(*)(void))THPVariable_contiguous, METH_VARARGS | METH_KEYWORDS, NULL},
  {"copy_", (PyCFunction)(void(*)(void))THPVariable_copy_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cpu", (PyCFunction)THPVariable_cpu, METH_NOARGS, NULL},
  {"cuda", (PyCFunction)(void(*)(void))THPVariable_cuda, METH_VARARGS | METH_KEYWORDS, NULL},
  {"data_ptr", (PyCFunction)THPVariable_data_ptr, METH_NOARGS, NULL},
  {"dim", (PyCFunction)THPVariable_dim, METH_NOARGS, NULL},
#ifdef BUILD_NAMEDTENSOR
  {"has_names", (PyCFunction)THPVariable_has_names, METH_NOARGS, NULL},
#endif
  {"double", (PyCFunction)THPVariable_double, METH_NOARGS, NULL},
  {"element_size", (PyCFunction)THPVariable_element_size, METH_NOARGS, NULL},
  {"float", (PyCFunction)THPVariable_float, METH_NOARGS, NULL},
  {"get_device", (PyCFunction)THPVariable_get_device, METH_NOARGS, NULL},
  {"bool", (PyCFunction)THPVariable_bool, METH_NOARGS, NULL},
  {"half", (PyCFunction)THPVariable_half, METH_NOARGS, NULL},
  {"int", (PyCFunction)THPVariable_int, METH_NOARGS, NULL},
  {"is_contiguous", (PyCFunction)(void(*)(void))THPVariable_is_contiguous, METH_VARARGS | METH_KEYWORDS, NULL},
  {"item", (PyCFunction)THPVariable_item, METH_NOARGS, NULL},
  {"long", (PyCFunction)THPVariable_long, METH_NOARGS, NULL},
  {"map_", (PyCFunction)(void(*)(void))THPVariable_map_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"map2_", (PyCFunction)(void(*)(void))THPVariable_map2_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ndimension", (PyCFunction)THPVariable_dim, METH_NOARGS, NULL},
  {"nelement", (PyCFunction)THPVariable_numel, METH_NOARGS, NULL},
  {"new", (PyCFunction)(void(*)(void))THPVariable_new, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_ones", (PyCFunction)(void(*)(void))THPVariable_new_ones, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_tensor", (PyCFunction)(void(*)(void))THPVariable_new_tensor, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_zeros", (PyCFunction)(void(*)(void))THPVariable_new_zeros, METH_VARARGS | METH_KEYWORDS, NULL},
  {"nonzero", (PyCFunction)(void(*)(void))THPVariable_nonzero, METH_VARARGS | METH_KEYWORDS, NULL},
  {"numpy", (PyCFunction)THPVariable_numpy, METH_NOARGS, NULL},
  {"record_stream", (PyCFunction)THPVariable_record_stream, METH_O, NULL},
  {"requires_grad_", (PyCFunction)(void(*)(void))THPVariable_requires_grad_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"short", (PyCFunction)THPVariable_short, METH_NOARGS, NULL},
  {"size", (PyCFunction)(void(*)(void))THPVariable_size, METH_VARARGS | METH_KEYWORDS, NULL},
  {"storage", (PyCFunction)THPVariable_storage, METH_NOARGS, NULL},
  {"storage_offset", (PyCFunction)THPVariable_storage_offset, METH_NOARGS, NULL},
  {"storage_type", (PyCFunction)THPVariable_storage_type, METH_NOARGS, NULL},
  {"stride", (PyCFunction)(void(*)(void))THPVariable_stride, METH_VARARGS | METH_KEYWORDS, NULL},
  {"to", (PyCFunction)(void(*)(void))THPVariable_to, METH_VARARGS | METH_KEYWORDS, NULL},
  {"tolist", (PyCFunction)THPVariable_tolist, METH_NOARGS, NULL},
  {"type", (PyCFunction)(void(*)(void))THPVariable_type, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__and__", (PyCFunction)(void(*)(void))THPVariable___and__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__iand__", (PyCFunction)(void(*)(void))THPVariable___iand__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__ilshift__", (PyCFunction)(void(*)(void))THPVariable___ilshift__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__ior__", (PyCFunction)(void(*)(void))THPVariable___ior__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__irshift__", (PyCFunction)(void(*)(void))THPVariable___irshift__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__ixor__", (PyCFunction)(void(*)(void))THPVariable___ixor__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__lshift__", (PyCFunction)(void(*)(void))THPVariable___lshift__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__or__", (PyCFunction)(void(*)(void))THPVariable___or__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__rshift__", (PyCFunction)(void(*)(void))THPVariable___rshift__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"__xor__", (PyCFunction)(void(*)(void))THPVariable___xor__, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_coalesced_", (PyCFunction)(void(*)(void))THPVariable__coalesced_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_dimI", (PyCFunction)THPVariable__dimI, METH_NOARGS, NULL},
  {"_dimV", (PyCFunction)THPVariable__dimV, METH_NOARGS, NULL},
  {"_indices", (PyCFunction)THPVariable__indices, METH_NOARGS, NULL},
  {"_nnz", (PyCFunction)THPVariable__nnz, METH_NOARGS, NULL},
  {"_values", (PyCFunction)THPVariable__values, METH_NOARGS, NULL},
  {"abs", (PyCFunction)THPVariable_abs, METH_NOARGS, NULL},
  {"abs_", (PyCFunction)THPVariable_abs_, METH_NOARGS, NULL},
  {"acos", (PyCFunction)THPVariable_acos, METH_NOARGS, NULL},
  {"acos_", (PyCFunction)THPVariable_acos_, METH_NOARGS, NULL},
  {"add", (PyCFunction)(void(*)(void))THPVariable_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"add_", (PyCFunction)(void(*)(void))THPVariable_add_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addbmm", (PyCFunction)(void(*)(void))THPVariable_addbmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addbmm_", (PyCFunction)(void(*)(void))THPVariable_addbmm_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addcdiv", (PyCFunction)(void(*)(void))THPVariable_addcdiv, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addcdiv_", (PyCFunction)(void(*)(void))THPVariable_addcdiv_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addcmul", (PyCFunction)(void(*)(void))THPVariable_addcmul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addcmul_", (PyCFunction)(void(*)(void))THPVariable_addcmul_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addmm", (PyCFunction)(void(*)(void))THPVariable_addmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addmm_", (PyCFunction)(void(*)(void))THPVariable_addmm_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addmv", (PyCFunction)(void(*)(void))THPVariable_addmv, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addmv_", (PyCFunction)(void(*)(void))THPVariable_addmv_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addr", (PyCFunction)(void(*)(void))THPVariable_addr, METH_VARARGS | METH_KEYWORDS, NULL},
  {"addr_", (PyCFunction)(void(*)(void))THPVariable_addr_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"align_as", (PyCFunction)(void(*)(void))THPVariable_align_as, METH_VARARGS | METH_KEYWORDS, NULL},
  {"align_to", (PyCFunction)(void(*)(void))THPVariable_align_to, METH_VARARGS | METH_KEYWORDS, NULL},
  {"all", (PyCFunction)(void(*)(void))THPVariable_all, METH_VARARGS | METH_KEYWORDS, NULL},
  {"allclose", (PyCFunction)(void(*)(void))THPVariable_allclose, METH_VARARGS | METH_KEYWORDS, NULL},
  {"any", (PyCFunction)(void(*)(void))THPVariable_any, METH_VARARGS | METH_KEYWORDS, NULL},
  {"argmax", (PyCFunction)(void(*)(void))THPVariable_argmax, METH_VARARGS | METH_KEYWORDS, NULL},
  {"argmin", (PyCFunction)(void(*)(void))THPVariable_argmin, METH_VARARGS | METH_KEYWORDS, NULL},
  {"argsort", (PyCFunction)(void(*)(void))THPVariable_argsort, METH_VARARGS | METH_KEYWORDS, NULL},
  {"as_strided", (PyCFunction)(void(*)(void))THPVariable_as_strided, METH_VARARGS | METH_KEYWORDS, NULL},
  {"as_strided_", (PyCFunction)(void(*)(void))THPVariable_as_strided_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"asin", (PyCFunction)THPVariable_asin, METH_NOARGS, NULL},
  {"asin_", (PyCFunction)THPVariable_asin_, METH_NOARGS, NULL},
  {"atan", (PyCFunction)THPVariable_atan, METH_NOARGS, NULL},
  {"atan2", (PyCFunction)(void(*)(void))THPVariable_atan2, METH_VARARGS | METH_KEYWORDS, NULL},
  {"atan2_", (PyCFunction)(void(*)(void))THPVariable_atan2_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"atan_", (PyCFunction)THPVariable_atan_, METH_NOARGS, NULL},
  {"backward", (PyCFunction)(void(*)(void))THPVariable_backward, METH_VARARGS | METH_KEYWORDS, NULL},
  {"baddbmm", (PyCFunction)(void(*)(void))THPVariable_baddbmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"baddbmm_", (PyCFunction)(void(*)(void))THPVariable_baddbmm_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bernoulli", (PyCFunction)(void(*)(void))THPVariable_bernoulli, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bernoulli_", (PyCFunction)(void(*)(void))THPVariable_bernoulli_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bincount", (PyCFunction)(void(*)(void))THPVariable_bincount, METH_VARARGS | METH_KEYWORDS, NULL},
  {"bitwise_not", (PyCFunction)THPVariable_bitwise_not, METH_NOARGS, NULL},
  {"bitwise_not_", (PyCFunction)THPVariable_bitwise_not_, METH_NOARGS, NULL},
  {"bmm", (PyCFunction)(void(*)(void))THPVariable_bmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cauchy_", (PyCFunction)(void(*)(void))THPVariable_cauchy_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ceil", (PyCFunction)THPVariable_ceil, METH_NOARGS, NULL},
  {"ceil_", (PyCFunction)THPVariable_ceil_, METH_NOARGS, NULL},
  {"cholesky", (PyCFunction)(void(*)(void))THPVariable_cholesky, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cholesky_inverse", (PyCFunction)(void(*)(void))THPVariable_cholesky_inverse, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cholesky_solve", (PyCFunction)(void(*)(void))THPVariable_cholesky_solve, METH_VARARGS | METH_KEYWORDS, NULL},
  {"chunk", (PyCFunction)(void(*)(void))THPVariable_chunk, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp", (PyCFunction)(void(*)(void))THPVariable_clamp, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp_", (PyCFunction)(void(*)(void))THPVariable_clamp_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp_max", (PyCFunction)(void(*)(void))THPVariable_clamp_max, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp_max_", (PyCFunction)(void(*)(void))THPVariable_clamp_max_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp_min", (PyCFunction)(void(*)(void))THPVariable_clamp_min, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clamp_min_", (PyCFunction)(void(*)(void))THPVariable_clamp_min_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"clone", (PyCFunction)THPVariable_clone, METH_NOARGS, NULL},
  {"coalesce", (PyCFunction)THPVariable_coalesce, METH_NOARGS, NULL},
  {"cos", (PyCFunction)THPVariable_cos, METH_NOARGS, NULL},
  {"cos_", (PyCFunction)THPVariable_cos_, METH_NOARGS, NULL},
  {"cosh", (PyCFunction)THPVariable_cosh, METH_NOARGS, NULL},
  {"cosh_", (PyCFunction)THPVariable_cosh_, METH_NOARGS, NULL},
  {"cross", (PyCFunction)(void(*)(void))THPVariable_cross, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cumprod", (PyCFunction)(void(*)(void))THPVariable_cumprod, METH_VARARGS | METH_KEYWORDS, NULL},
  {"cumsum", (PyCFunction)(void(*)(void))THPVariable_cumsum, METH_VARARGS | METH_KEYWORDS, NULL},
  {"dense_dim", (PyCFunction)THPVariable_dense_dim, METH_NOARGS, NULL},
  {"dequantize", (PyCFunction)THPVariable_dequantize, METH_NOARGS, NULL},
  {"det", (PyCFunction)THPVariable_det, METH_NOARGS, NULL},
  {"detach", (PyCFunction)THPVariable_detach, METH_NOARGS, NULL},
  {"detach_", (PyCFunction)THPVariable_detach_, METH_NOARGS, NULL},
  {"diag", (PyCFunction)(void(*)(void))THPVariable_diag, METH_VARARGS | METH_KEYWORDS, NULL},
  {"diag_embed", (PyCFunction)(void(*)(void))THPVariable_diag_embed, METH_VARARGS | METH_KEYWORDS, NULL},
  {"diagflat", (PyCFunction)(void(*)(void))THPVariable_diagflat, METH_VARARGS | METH_KEYWORDS, NULL},
  {"diagonal", (PyCFunction)(void(*)(void))THPVariable_diagonal, METH_VARARGS | METH_KEYWORDS, NULL},
  {"digamma", (PyCFunction)THPVariable_digamma, METH_NOARGS, NULL},
  {"digamma_", (PyCFunction)THPVariable_digamma_, METH_NOARGS, NULL},
  {"dist", (PyCFunction)(void(*)(void))THPVariable_dist, METH_VARARGS | METH_KEYWORDS, NULL},
  {"div", (PyCFunction)(void(*)(void))THPVariable_div, METH_VARARGS | METH_KEYWORDS, NULL},
  {"div_", (PyCFunction)(void(*)(void))THPVariable_div_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"dot", (PyCFunction)(void(*)(void))THPVariable_dot, METH_VARARGS | METH_KEYWORDS, NULL},
  {"eig", (PyCFunction)(void(*)(void))THPVariable_eig, METH_VARARGS | METH_KEYWORDS, NULL},
  {"eq", (PyCFunction)(void(*)(void))THPVariable_eq, METH_VARARGS | METH_KEYWORDS, NULL},
  {"eq_", (PyCFunction)(void(*)(void))THPVariable_eq_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"equal", (PyCFunction)(void(*)(void))THPVariable_equal, METH_VARARGS | METH_KEYWORDS, NULL},
  {"erf", (PyCFunction)THPVariable_erf, METH_NOARGS, NULL},
  {"erf_", (PyCFunction)THPVariable_erf_, METH_NOARGS, NULL},
  {"erfc", (PyCFunction)THPVariable_erfc, METH_NOARGS, NULL},
  {"erfc_", (PyCFunction)THPVariable_erfc_, METH_NOARGS, NULL},
  {"erfinv", (PyCFunction)THPVariable_erfinv, METH_NOARGS, NULL},
  {"erfinv_", (PyCFunction)THPVariable_erfinv_, METH_NOARGS, NULL},
  {"exp", (PyCFunction)THPVariable_exp, METH_NOARGS, NULL},
  {"exp_", (PyCFunction)THPVariable_exp_, METH_NOARGS, NULL},
  {"expand", (PyCFunction)(void(*)(void))THPVariable_expand, METH_VARARGS | METH_KEYWORDS, NULL},
  {"expand_as", (PyCFunction)(void(*)(void))THPVariable_expand_as, METH_VARARGS | METH_KEYWORDS, NULL},
  {"expm1", (PyCFunction)THPVariable_expm1, METH_NOARGS, NULL},
  {"expm1_", (PyCFunction)THPVariable_expm1_, METH_NOARGS, NULL},
  {"exponential_", (PyCFunction)(void(*)(void))THPVariable_exponential_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"fft", (PyCFunction)(void(*)(void))THPVariable_fft, METH_VARARGS | METH_KEYWORDS, NULL},
  {"fill_", (PyCFunction)(void(*)(void))THPVariable_fill_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"fill_diagonal_", (PyCFunction)(void(*)(void))THPVariable_fill_diagonal_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"flatten", (PyCFunction)(void(*)(void))THPVariable_flatten, METH_VARARGS | METH_KEYWORDS, NULL},
  {"flip", (PyCFunction)(void(*)(void))THPVariable_flip, METH_VARARGS | METH_KEYWORDS, NULL},
  {"floor", (PyCFunction)THPVariable_floor, METH_NOARGS, NULL},
  {"floor_", (PyCFunction)THPVariable_floor_, METH_NOARGS, NULL},
  {"fmod", (PyCFunction)(void(*)(void))THPVariable_fmod, METH_VARARGS | METH_KEYWORDS, NULL},
  {"fmod_", (PyCFunction)(void(*)(void))THPVariable_fmod_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"frac", (PyCFunction)THPVariable_frac, METH_NOARGS, NULL},
  {"frac_", (PyCFunction)THPVariable_frac_, METH_NOARGS, NULL},
  {"gather", (PyCFunction)(void(*)(void))THPVariable_gather, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ge", (PyCFunction)(void(*)(void))THPVariable_ge, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ge_", (PyCFunction)(void(*)(void))THPVariable_ge_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"geometric_", (PyCFunction)(void(*)(void))THPVariable_geometric_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"geqrf", (PyCFunction)THPVariable_geqrf, METH_NOARGS, NULL},
  {"ger", (PyCFunction)(void(*)(void))THPVariable_ger, METH_VARARGS | METH_KEYWORDS, NULL},
  {"gt", (PyCFunction)(void(*)(void))THPVariable_gt, METH_VARARGS | METH_KEYWORDS, NULL},
  {"gt_", (PyCFunction)(void(*)(void))THPVariable_gt_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"hardshrink", (PyCFunction)(void(*)(void))THPVariable_hardshrink, METH_VARARGS | METH_KEYWORDS, NULL},
  {"histc", (PyCFunction)(void(*)(void))THPVariable_histc, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ifft", (PyCFunction)(void(*)(void))THPVariable_ifft, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_add", (PyCFunction)(void(*)(void))THPVariable_index_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_add_", (PyCFunction)(void(*)(void))THPVariable_index_add_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_copy", (PyCFunction)(void(*)(void))THPVariable_index_copy, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_copy_", (PyCFunction)(void(*)(void))THPVariable_index_copy_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_fill", (PyCFunction)(void(*)(void))THPVariable_index_fill, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_fill_", (PyCFunction)(void(*)(void))THPVariable_index_fill_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_put", (PyCFunction)(void(*)(void))THPVariable_index_put, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_put_", (PyCFunction)(void(*)(void))THPVariable_index_put_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"index_select", (PyCFunction)(void(*)(void))THPVariable_index_select, METH_VARARGS | METH_KEYWORDS, NULL},
  {"indices", (PyCFunction)THPVariable_indices, METH_NOARGS, NULL},
  {"int_repr", (PyCFunction)THPVariable_int_repr, METH_NOARGS, NULL},
  {"inverse", (PyCFunction)THPVariable_inverse, METH_NOARGS, NULL},
  {"irfft", (PyCFunction)(void(*)(void))THPVariable_irfft, METH_VARARGS | METH_KEYWORDS, NULL},
  {"is_coalesced", (PyCFunction)THPVariable_is_coalesced, METH_NOARGS, NULL},
  {"is_complex", (PyCFunction)THPVariable_is_complex, METH_NOARGS, NULL},
  {"is_distributed", (PyCFunction)THPVariable_is_distributed, METH_NOARGS, NULL},
  {"is_floating_point", (PyCFunction)THPVariable_is_floating_point, METH_NOARGS, NULL},
  {"is_nonzero", (PyCFunction)THPVariable_is_nonzero, METH_NOARGS, NULL},
  {"is_pinned", (PyCFunction)THPVariable_is_pinned, METH_NOARGS, NULL},
  {"is_same_size", (PyCFunction)(void(*)(void))THPVariable_is_same_size, METH_VARARGS | METH_KEYWORDS, NULL},
  {"is_set_to", (PyCFunction)(void(*)(void))THPVariable_is_set_to, METH_VARARGS | METH_KEYWORDS, NULL},
  {"is_signed", (PyCFunction)THPVariable_is_signed, METH_NOARGS, NULL},
  {"isclose", (PyCFunction)(void(*)(void))THPVariable_isclose, METH_VARARGS | METH_KEYWORDS, NULL},
  {"kthvalue", (PyCFunction)(void(*)(void))THPVariable_kthvalue, METH_VARARGS | METH_KEYWORDS, NULL},
  {"le", (PyCFunction)(void(*)(void))THPVariable_le, METH_VARARGS | METH_KEYWORDS, NULL},
  {"le_", (PyCFunction)(void(*)(void))THPVariable_le_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lerp", (PyCFunction)(void(*)(void))THPVariable_lerp, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lerp_", (PyCFunction)(void(*)(void))THPVariable_lerp_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lgamma", (PyCFunction)THPVariable_lgamma, METH_NOARGS, NULL},
  {"lgamma_", (PyCFunction)THPVariable_lgamma_, METH_NOARGS, NULL},
  {"log", (PyCFunction)THPVariable_log, METH_NOARGS, NULL},
  {"log10", (PyCFunction)THPVariable_log10, METH_NOARGS, NULL},
  {"log10_", (PyCFunction)THPVariable_log10_, METH_NOARGS, NULL},
  {"log1p", (PyCFunction)THPVariable_log1p, METH_NOARGS, NULL},
  {"log1p_", (PyCFunction)THPVariable_log1p_, METH_NOARGS, NULL},
  {"log2", (PyCFunction)THPVariable_log2, METH_NOARGS, NULL},
  {"log2_", (PyCFunction)THPVariable_log2_, METH_NOARGS, NULL},
  {"log_", (PyCFunction)THPVariable_log_, METH_NOARGS, NULL},
  {"log_normal_", (PyCFunction)(void(*)(void))THPVariable_log_normal_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"log_softmax", (PyCFunction)(void(*)(void))THPVariable_log_softmax, METH_VARARGS | METH_KEYWORDS, NULL},
  {"logdet", (PyCFunction)THPVariable_logdet, METH_NOARGS, NULL},
  {"logical_not", (PyCFunction)THPVariable_logical_not, METH_NOARGS, NULL},
  {"logical_not_", (PyCFunction)THPVariable_logical_not_, METH_NOARGS, NULL},
  {"logical_xor", (PyCFunction)(void(*)(void))THPVariable_logical_xor, METH_VARARGS | METH_KEYWORDS, NULL},
  {"logical_xor_", (PyCFunction)(void(*)(void))THPVariable_logical_xor_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"logsumexp", (PyCFunction)(void(*)(void))THPVariable_logsumexp, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lstsq", (PyCFunction)(void(*)(void))THPVariable_lstsq, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lt", (PyCFunction)(void(*)(void))THPVariable_lt, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lt_", (PyCFunction)(void(*)(void))THPVariable_lt_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"lu_solve", (PyCFunction)(void(*)(void))THPVariable_lu_solve, METH_VARARGS | METH_KEYWORDS, NULL},
  {"masked_fill", (PyCFunction)(void(*)(void))THPVariable_masked_fill, METH_VARARGS | METH_KEYWORDS, NULL},
  {"masked_fill_", (PyCFunction)(void(*)(void))THPVariable_masked_fill_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"masked_scatter", (PyCFunction)(void(*)(void))THPVariable_masked_scatter, METH_VARARGS | METH_KEYWORDS, NULL},
  {"masked_scatter_", (PyCFunction)(void(*)(void))THPVariable_masked_scatter_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"masked_select", (PyCFunction)(void(*)(void))THPVariable_masked_select, METH_VARARGS | METH_KEYWORDS, NULL},
  {"matmul", (PyCFunction)(void(*)(void))THPVariable_matmul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"matrix_power", (PyCFunction)(void(*)(void))THPVariable_matrix_power, METH_VARARGS | METH_KEYWORDS, NULL},
  {"max", (PyCFunction)(void(*)(void))THPVariable_max, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mean", (PyCFunction)(void(*)(void))THPVariable_mean, METH_VARARGS | METH_KEYWORDS, NULL},
  {"median", (PyCFunction)(void(*)(void))THPVariable_median, METH_VARARGS | METH_KEYWORDS, NULL},
  {"min", (PyCFunction)(void(*)(void))THPVariable_min, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mm", (PyCFunction)(void(*)(void))THPVariable_mm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mode", (PyCFunction)(void(*)(void))THPVariable_mode, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mul", (PyCFunction)(void(*)(void))THPVariable_mul, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mul_", (PyCFunction)(void(*)(void))THPVariable_mul_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"multinomial", (PyCFunction)(void(*)(void))THPVariable_multinomial, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mv", (PyCFunction)(void(*)(void))THPVariable_mv, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mvlgamma", (PyCFunction)(void(*)(void))THPVariable_mvlgamma, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mvlgamma_", (PyCFunction)(void(*)(void))THPVariable_mvlgamma_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"narrow", (PyCFunction)(void(*)(void))THPVariable_narrow, METH_VARARGS | METH_KEYWORDS, NULL},
  {"narrow_copy", (PyCFunction)(void(*)(void))THPVariable_narrow_copy, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ne", (PyCFunction)(void(*)(void))THPVariable_ne, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ne_", (PyCFunction)(void(*)(void))THPVariable_ne_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"neg", (PyCFunction)THPVariable_neg, METH_NOARGS, NULL},
  {"neg_", (PyCFunction)THPVariable_neg_, METH_NOARGS, NULL},
  {"new_empty", (PyCFunction)(void(*)(void))THPVariable_new_empty, METH_VARARGS | METH_KEYWORDS, NULL},
  {"new_full", (PyCFunction)(void(*)(void))THPVariable_new_full, METH_VARARGS | METH_KEYWORDS, NULL},
  {"norm", (PyCFunction)(void(*)(void))THPVariable_norm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"normal_", (PyCFunction)(void(*)(void))THPVariable_normal_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"numel", (PyCFunction)THPVariable_numel, METH_NOARGS, NULL},
  {"orgqr", (PyCFunction)(void(*)(void))THPVariable_orgqr, METH_VARARGS | METH_KEYWORDS, NULL},
  {"ormqr", (PyCFunction)(void(*)(void))THPVariable_ormqr, METH_VARARGS | METH_KEYWORDS, NULL},
  {"permute", (PyCFunction)(void(*)(void))THPVariable_permute, METH_VARARGS | METH_KEYWORDS, NULL},
  {"pin_memory", (PyCFunction)THPVariable_pin_memory, METH_NOARGS, NULL},
  {"pinverse", (PyCFunction)(void(*)(void))THPVariable_pinverse, METH_VARARGS | METH_KEYWORDS, NULL},
  {"polygamma", (PyCFunction)(void(*)(void))THPVariable_polygamma, METH_VARARGS | METH_KEYWORDS, NULL},
  {"polygamma_", (PyCFunction)(void(*)(void))THPVariable_polygamma_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"pow", (PyCFunction)(void(*)(void))THPVariable_pow, METH_VARARGS | METH_KEYWORDS, NULL},
  {"pow_", (PyCFunction)(void(*)(void))THPVariable_pow_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"prelu", (PyCFunction)(void(*)(void))THPVariable_prelu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"prod", (PyCFunction)(void(*)(void))THPVariable_prod, METH_VARARGS | METH_KEYWORDS, NULL},
  {"put_", (PyCFunction)(void(*)(void))THPVariable_put_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"q_per_channel_axis", (PyCFunction)THPVariable_q_per_channel_axis, METH_NOARGS, NULL},
  {"q_per_channel_scales", (PyCFunction)THPVariable_q_per_channel_scales, METH_NOARGS, NULL},
  {"q_per_channel_zero_points", (PyCFunction)THPVariable_q_per_channel_zero_points, METH_NOARGS, NULL},
  {"q_scale", (PyCFunction)THPVariable_q_scale, METH_NOARGS, NULL},
  {"q_zero_point", (PyCFunction)THPVariable_q_zero_point, METH_NOARGS, NULL},
  {"qr", (PyCFunction)(void(*)(void))THPVariable_qr, METH_VARARGS | METH_KEYWORDS, NULL},
  {"qscheme", (PyCFunction)THPVariable_qscheme, METH_NOARGS, NULL},
  {"random_", (PyCFunction)(void(*)(void))THPVariable_random_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"reciprocal", (PyCFunction)THPVariable_reciprocal, METH_NOARGS, NULL},
  {"reciprocal_", (PyCFunction)THPVariable_reciprocal_, METH_NOARGS, NULL},
  {"refine_names", (PyCFunction)(void(*)(void))THPVariable_refine_names, METH_VARARGS | METH_KEYWORDS, NULL},
  {"relu", (PyCFunction)THPVariable_relu, METH_NOARGS, NULL},
  {"relu_", (PyCFunction)THPVariable_relu_, METH_NOARGS, NULL},
  {"remainder", (PyCFunction)(void(*)(void))THPVariable_remainder, METH_VARARGS | METH_KEYWORDS, NULL},
  {"remainder_", (PyCFunction)(void(*)(void))THPVariable_remainder_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"rename", (PyCFunction)(void(*)(void))THPVariable_rename, METH_VARARGS | METH_KEYWORDS, NULL},
  {"rename_", (PyCFunction)(void(*)(void))THPVariable_rename_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"renorm", (PyCFunction)(void(*)(void))THPVariable_renorm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"renorm_", (PyCFunction)(void(*)(void))THPVariable_renorm_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"repeat", (PyCFunction)(void(*)(void))THPVariable_repeat, METH_VARARGS | METH_KEYWORDS, NULL},
  {"repeat_interleave", (PyCFunction)(void(*)(void))THPVariable_repeat_interleave, METH_VARARGS | METH_KEYWORDS, NULL},
  {"reshape", (PyCFunction)(void(*)(void))THPVariable_reshape, METH_VARARGS | METH_KEYWORDS, NULL},
  {"reshape_as", (PyCFunction)(void(*)(void))THPVariable_reshape_as, METH_VARARGS | METH_KEYWORDS, NULL},
  {"resize_", (PyCFunction)(void(*)(void))THPVariable_resize_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"resize_as_", (PyCFunction)(void(*)(void))THPVariable_resize_as_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"rfft", (PyCFunction)(void(*)(void))THPVariable_rfft, METH_VARARGS | METH_KEYWORDS, NULL},
  {"roll", (PyCFunction)(void(*)(void))THPVariable_roll, METH_VARARGS | METH_KEYWORDS, NULL},
  {"rot90", (PyCFunction)(void(*)(void))THPVariable_rot90, METH_VARARGS | METH_KEYWORDS, NULL},
  {"round", (PyCFunction)THPVariable_round, METH_NOARGS, NULL},
  {"round_", (PyCFunction)THPVariable_round_, METH_NOARGS, NULL},
  {"rsqrt", (PyCFunction)THPVariable_rsqrt, METH_NOARGS, NULL},
  {"rsqrt_", (PyCFunction)THPVariable_rsqrt_, METH_NOARGS, NULL},
  {"scatter", (PyCFunction)(void(*)(void))THPVariable_scatter, METH_VARARGS | METH_KEYWORDS, NULL},
  {"scatter_", (PyCFunction)(void(*)(void))THPVariable_scatter_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"scatter_add", (PyCFunction)(void(*)(void))THPVariable_scatter_add, METH_VARARGS | METH_KEYWORDS, NULL},
  {"scatter_add_", (PyCFunction)(void(*)(void))THPVariable_scatter_add_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"select", (PyCFunction)(void(*)(void))THPVariable_select, METH_VARARGS | METH_KEYWORDS, NULL},
  {"set_", (PyCFunction)(void(*)(void))THPVariable_set_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sigmoid", (PyCFunction)THPVariable_sigmoid, METH_NOARGS, NULL},
  {"sigmoid_", (PyCFunction)THPVariable_sigmoid_, METH_NOARGS, NULL},
  {"sign", (PyCFunction)THPVariable_sign, METH_NOARGS, NULL},
  {"sign_", (PyCFunction)THPVariable_sign_, METH_NOARGS, NULL},
  {"sin", (PyCFunction)THPVariable_sin, METH_NOARGS, NULL},
  {"sin_", (PyCFunction)THPVariable_sin_, METH_NOARGS, NULL},
  {"sinh", (PyCFunction)THPVariable_sinh, METH_NOARGS, NULL},
  {"sinh_", (PyCFunction)THPVariable_sinh_, METH_NOARGS, NULL},
  {"slogdet", (PyCFunction)THPVariable_slogdet, METH_NOARGS, NULL},
  {"smm", (PyCFunction)(void(*)(void))THPVariable_smm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"softmax", (PyCFunction)(void(*)(void))THPVariable_softmax, METH_VARARGS | METH_KEYWORDS, NULL},
  {"solve", (PyCFunction)(void(*)(void))THPVariable_solve, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sort", (PyCFunction)(void(*)(void))THPVariable_sort, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sparse_dim", (PyCFunction)THPVariable_sparse_dim, METH_NOARGS, NULL},
  {"sparse_mask", (PyCFunction)(void(*)(void))THPVariable_sparse_mask, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sparse_resize_", (PyCFunction)(void(*)(void))THPVariable_sparse_resize_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sparse_resize_and_clear_", (PyCFunction)(void(*)(void))THPVariable_sparse_resize_and_clear_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"split", (PyCFunction)(void(*)(void))THPVariable_split, METH_VARARGS | METH_KEYWORDS, NULL},
  {"split_with_sizes", (PyCFunction)(void(*)(void))THPVariable_split_with_sizes, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sqrt", (PyCFunction)THPVariable_sqrt, METH_NOARGS, NULL},
  {"sqrt_", (PyCFunction)THPVariable_sqrt_, METH_NOARGS, NULL},
  {"squeeze", (PyCFunction)(void(*)(void))THPVariable_squeeze, METH_VARARGS | METH_KEYWORDS, NULL},
  {"squeeze_", (PyCFunction)(void(*)(void))THPVariable_squeeze_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sspaddmm", (PyCFunction)(void(*)(void))THPVariable_sspaddmm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"std", (PyCFunction)(void(*)(void))THPVariable_std, METH_VARARGS | METH_KEYWORDS, NULL},
  {"stft", (PyCFunction)(void(*)(void))THPVariable_stft, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sub", (PyCFunction)(void(*)(void))THPVariable_sub, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sub_", (PyCFunction)(void(*)(void))THPVariable_sub_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sum", (PyCFunction)(void(*)(void))THPVariable_sum, METH_VARARGS | METH_KEYWORDS, NULL},
  {"sum_to_size", (PyCFunction)(void(*)(void))THPVariable_sum_to_size, METH_VARARGS | METH_KEYWORDS, NULL},
  {"svd", (PyCFunction)(void(*)(void))THPVariable_svd, METH_VARARGS | METH_KEYWORDS, NULL},
  {"symeig", (PyCFunction)(void(*)(void))THPVariable_symeig, METH_VARARGS | METH_KEYWORDS, NULL},
  {"t", (PyCFunction)THPVariable_t, METH_NOARGS, NULL},
  {"t_", (PyCFunction)THPVariable_t_, METH_NOARGS, NULL},
  {"take", (PyCFunction)(void(*)(void))THPVariable_take, METH_VARARGS | METH_KEYWORDS, NULL},
  {"tan", (PyCFunction)THPVariable_tan, METH_NOARGS, NULL},
  {"tan_", (PyCFunction)THPVariable_tan_, METH_NOARGS, NULL},
  {"tanh", (PyCFunction)THPVariable_tanh, METH_NOARGS, NULL},
  {"tanh_", (PyCFunction)THPVariable_tanh_, METH_NOARGS, NULL},
  {"to_dense", (PyCFunction)THPVariable_to_dense, METH_NOARGS, NULL},
  {"to_mkldnn", (PyCFunction)THPVariable_to_mkldnn, METH_NOARGS, NULL},
  {"to_sparse", (PyCFunction)(void(*)(void))THPVariable_to_sparse, METH_VARARGS | METH_KEYWORDS, NULL},
  {"topk", (PyCFunction)(void(*)(void))THPVariable_topk, METH_VARARGS | METH_KEYWORDS, NULL},
  {"trace", (PyCFunction)THPVariable_trace, METH_NOARGS, NULL},
  {"transpose", (PyCFunction)(void(*)(void))THPVariable_transpose, METH_VARARGS | METH_KEYWORDS, NULL},
  {"transpose_", (PyCFunction)(void(*)(void))THPVariable_transpose_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"triangular_solve", (PyCFunction)(void(*)(void))THPVariable_triangular_solve, METH_VARARGS | METH_KEYWORDS, NULL},
  {"tril", (PyCFunction)(void(*)(void))THPVariable_tril, METH_VARARGS | METH_KEYWORDS, NULL},
  {"tril_", (PyCFunction)(void(*)(void))THPVariable_tril_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"triu", (PyCFunction)(void(*)(void))THPVariable_triu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"triu_", (PyCFunction)(void(*)(void))THPVariable_triu_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"trunc", (PyCFunction)THPVariable_trunc, METH_NOARGS, NULL},
  {"trunc_", (PyCFunction)THPVariable_trunc_, METH_NOARGS, NULL},
  {"type_as", (PyCFunction)(void(*)(void))THPVariable_type_as, METH_VARARGS | METH_KEYWORDS, NULL},
  {"unbind", (PyCFunction)(void(*)(void))THPVariable_unbind, METH_VARARGS | METH_KEYWORDS, NULL},
  {"unflatten", (PyCFunction)(void(*)(void))THPVariable_unflatten, METH_VARARGS | METH_KEYWORDS, NULL},
  {"unfold", (PyCFunction)(void(*)(void))THPVariable_unfold, METH_VARARGS | METH_KEYWORDS, NULL},
  {"uniform_", (PyCFunction)(void(*)(void))THPVariable_uniform_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"unsqueeze", (PyCFunction)(void(*)(void))THPVariable_unsqueeze, METH_VARARGS | METH_KEYWORDS, NULL},
  {"unsqueeze_", (PyCFunction)(void(*)(void))THPVariable_unsqueeze_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"values", (PyCFunction)THPVariable_values, METH_NOARGS, NULL},
  {"var", (PyCFunction)(void(*)(void))THPVariable_var, METH_VARARGS | METH_KEYWORDS, NULL},
  {"view", (PyCFunction)(void(*)(void))THPVariable_view, METH_VARARGS | METH_KEYWORDS, NULL},
  {"view_as", (PyCFunction)(void(*)(void))THPVariable_view_as, METH_VARARGS | METH_KEYWORDS, NULL},
  {"where", (PyCFunction)(void(*)(void))THPVariable_where, METH_VARARGS | METH_KEYWORDS, NULL},
  {"zero_", (PyCFunction)THPVariable_zero_, METH_NOARGS, NULL},
  {NULL}
};

}} // namespace torch::autograd
