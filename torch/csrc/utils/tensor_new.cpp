#include "torch/csrc/python_headers.h"
#include "tensor_new.h"

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Size.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/cuda_lazy_init.h"
#include "torch/csrc/utils/numpy_stub.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/python_scalars.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/tensor_conversion_dispatch.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <c10/util/Exception.h>
#include "c10/util/Optional.h"

#include <stdexcept>
#include <vector>

using at::Backend;
using at::Device;
using at::IntList;
using at::kCPU;
using at::kCUDA;
using at::kLong;
using at::Scalar;
using at::ScalarType;
using at::Storage;
using at::Tensor;
using at::TensorOptions;
using at::Type;
using c10::optional;

namespace torch { namespace utils {
namespace {
const int MAX_DIMS = 128;

void maybe_initialize_cuda(const Type &type) {
  if (type.is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
}

Tensor dispatch_zeros(const Type& type, int32_t device_index, IntList sizes) {
  maybe_initialize_cuda(type);
  AutoNoGIL no_gil;
  return torch::zeros(sizes, type.options(device_index));
}

Tensor dispatch_ones(const Type& type, int32_t device_index, IntList sizes) {
  maybe_initialize_cuda(type);
  AutoNoGIL no_gil;
  return torch::ones(sizes, type.options(device_index));
}

Tensor dispatch_full(const Type& type, Scalar fill_value, int32_t device_index, IntList sizes) {
  maybe_initialize_cuda(type);
  AutoNoGIL no_gil;
  return torch::full(sizes, fill_value, type.options(device_index));
}

Tensor new_with_sizes(const Type& type, int32_t device_index, IntList sizes) {
  maybe_initialize_cuda(type);
  AutoNoGIL no_gil;
  return torch::empty(sizes, type.options(device_index));
}

Tensor new_with_storage(const Type& type, Storage storage) {
  auto tensor = at::empty({}, type.options());
  tensor.set_(storage);
  return tensor;
}

Tensor new_with_tensor(const Type& type, Tensor other) {
  if (other.type() != type) {
    throw TypeError("expected %s (got %s)", type.toString(), other.type().toString());
  }
  return other.slice();
}

Tensor new_with_type_conversion(const Type& type, Tensor other, int32_t device_index) {
  return dispatch_type_conversion(other, type, device_index, false);
}

Tensor new_with_tensor_copy(const Type& type, Tensor other, int32_t device_index) {
  maybe_initialize_cuda(type);
  AutoNoGIL no_gil;
  // TODO: It would be better if new_with_tensor_copy took an at::Device
  // to begin with, but then we need to fix the situation with
  // dispatch_type_conversion bleggg
  at::OptionalDeviceGuard device_guard;
  if (type.is_cuda()) {
    device_guard.reset_device(at::Device(at::kCUDA, device_index));
  }
  return type.copy(other);
}

std::vector<int64_t> compute_sizes(PyObject* seq) {
  std::vector<int64_t> sizes;
  THPObjectPtr handle;
  while (PySequence_Check(seq)) {
    auto length = PySequence_Length(seq);
    if (length < 0) throw python_error();
    sizes.push_back(length);
    if (sizes.size() > MAX_DIMS) {
      throw ValueError("too many dimensions '%s'", Py_TYPE(seq)->tp_name);
    }
    if (length == 0) break;
    handle = THPObjectPtr(PySequence_GetItem(seq, 0));
    if (!handle) {
      throw ValueError("could not determine the shape of object type '%s'", Py_TYPE(seq)->tp_name);
    }
    seq = handle.get();
  }

  return sizes;
}

ScalarType infer_scalar_type(PyObject *obj) {
  if (PyFloat_Check(obj)) {
    // this is always guaranteed to be a floating-point type, and makes it more
    // convenient to write e.g. torch.tensor(0.) than torch.tensor(0., dtype=torch.Tensor.dtype).
    return torch::tensors::get_default_tensor_type().scalarType();
  }
  if (THPUtils_checkLong(obj)) {
    return ScalarType::Long;
  }
  if (PyBool_Check(obj)) {
    // TODO: infer Bool when we have Bool ScalarType
    return ScalarType::Byte;
  }
  if (THPVariable_Check(obj)) {
    auto var = reinterpret_cast<THPVariable*>(obj)->cdata;
    return var.type().scalarType();
  }
#ifdef USE_NUMPY
  if (PyArray_Check(obj)) {
    return numpy_dtype_to_aten(PyArray_TYPE((PyArrayObject*)obj));
  }
  if (PyArray_CheckScalar(obj)) {
    return numpy_dtype_to_aten(PyArray_TYPE((PyArrayObject*)(PyArray_FromScalar(obj, nullptr))));
  }
#endif
  if (THPUtils_checkString(obj)) {
    throw TypeError("new(): invalid data type '%s'", Py_TYPE(obj)->tp_name);
  }
  if (PySequence_Check(obj)) {
    c10::optional<ScalarType> scalarType;
    auto length = PySequence_Length(obj);
    if (length < 0) throw python_error();
    // match NumPy semantics, except use default tensor type instead of double.
    if (length == 0) return torch::tensors::get_default_tensor_type().scalarType();
    for (int i = 0; i < length; ++i) {
      THPObjectPtr handle(PySequence_GetItem(obj, i));
      if (!handle) throw python_error();
      auto cur_item = handle.get();
      if (cur_item == obj) throw TypeError("new(): self-referential lists are incompatible");
      ScalarType item_scalarType = infer_scalar_type(cur_item);
      scalarType = (scalarType) ?
          at::promoteTypes(*scalarType, item_scalarType) : item_scalarType;
      if (scalarType == ScalarType::Double) {
        // this won't change (unless we hit undefined, but that will fail later).
        return *scalarType;
      }
    }
    return *scalarType;
  }
  AT_ERROR("Could not infer dtype of ", Py_TYPE(obj)->tp_name);
}

void recursive_store(char* data, IntList sizes, IntList strides, int64_t dim,
                            ScalarType scalarType, int elementSize, PyObject* obj) {
  int64_t ndim = sizes.size();
  if (dim == ndim) {
    torch::utils::store_scalar(data, scalarType, obj);
    return;
  }

  auto n = sizes[dim];
  auto seq = THPObjectPtr(PySequence_Fast(obj, "not a sequence"));
  if (!seq) throw python_error();
  auto seq_size = PySequence_Fast_GET_SIZE(seq.get());
  if (seq_size != n) {
    throw ValueError("expected sequence of length %lld at dim %lld (got %lld)",
      (long long)n, (long long)dim, (long long)seq_size);
  }

  PyObject** items = PySequence_Fast_ITEMS(seq.get());
  for (int64_t i = 0; i < n; i++) {
    recursive_store(data, sizes, strides, dim + 1, scalarType, elementSize, items[i]);
    data += strides[dim] * elementSize;
  }
}

Tensor internal_new_from_data(
    const Type& type,
    c10::optional<Device> device_opt,
    PyObject* data,
    bool copy_variables,
    bool copy_numpy,
    bool type_inference) {
  int32_t device_index = -1;
  if (device_opt.has_value()) {
    device_index = device_opt->index();
  }
  if (THPUtils_checkString(data)) {
    throw TypeError("new(): invalid data type '%s'", Py_TYPE(data)->tp_name);
  }

  if (THPVariable_Check(data)) {
    auto var = reinterpret_cast<THPVariable*>(data)->cdata;
    auto type_inference_device_type = device_opt.has_value() ? device_opt->type()
                                                             : torch::getDeviceType(var.type());
    // infer the scalar type and device type; it's not expected to infer the layout since these constructors
    // are defined per-layout-type (e.g. tensor vs sparse_coo_tensor).
    const auto& type_inference_type = torch::getVariableType(var.type().scalarType(),
                                                     *torch::getLayout(type.backend()),
                                                     type_inference_device_type);
    const auto& type_to_use = type_inference ? type_inference_type : type;
    return copy_variables ? new_with_tensor_copy(type_to_use, var, device_index)
                          : new_with_type_conversion(type_to_use, var, device_index);
  }

#ifdef USE_NUMPY
  if (PyArray_Check(data)) {
    auto tensor = autograd::make_variable(tensor_from_numpy(data), /*requires_grad=*/false);
    const auto& type_to_use = type_inference ? type.toScalarType(tensor.type().scalarType()) : type;
    return copy_numpy ? new_with_tensor_copy(type_to_use, tensor, device_index) :
                        new_with_type_conversion(type_to_use, tensor, device_index);
  }
#endif

  auto sizes = compute_sizes(data);
  ScalarType scalarType = type_inference ? infer_scalar_type(data) : type.scalarType();
  auto tensor = autograd::make_variable(at::empty(sizes, at::initialTensorOptions().dtype(scalarType)), /*requires_grad=*/false);
  recursive_store(
      (char*)tensor.data_ptr(), tensor.sizes(), tensor.strides(), 0,
      scalarType, tensor.type().elementSizeInBytes(), data);
  const auto& type_to_use = type_inference ? type.toScalarType(scalarType) : type;
  return new_with_type_conversion(type_to_use, tensor, device_index);
}

Tensor new_from_data_copy(
    const Type& type,
    c10::optional<Device> device,
    PyObject* data) {
  return internal_new_from_data(type, device, data, true, true, false);
}

Tensor legacy_new_from_sequence(
    const Type& type,
    c10::optional<Device> device,
    PyObject* data) {
  if (!PySequence_Check(data)) {
    throw TypeError("new(): data must be a sequence (got %s)", Py_TYPE(data)->tp_name);
  }
  return legacy_new_from_data(type, device, data);
}

void check_legacy_ctor_device(const Type& type, c10::optional<Device> device) {
  if (device.has_value()) {
    AT_CHECK(type.device_type() == device.value().type(),
             "legacy constructor for device type: ", type.device_type(),
             " was passed device type: ", device.value().type(),
             ", but device type must be: ", type.device_type());
  }
}

Tensor legacy_sparse_tensor_ctor(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, Device? device=None)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor indices, Tensor values, *, Device? device=None)",
    "new(Tensor indices, Tensor values, IntList size, *, Device? device=None)",
    "new(IntList size, *, Device? device=None)",
  });
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto deviceOptional = r.deviceOptional(0);
    check_legacy_ctor_device(type, deviceOptional);
    return at::empty({0}, type.options(r.device(0).index()));
  } else if (r.idx == 1) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return type.unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 2) {
    auto deviceOptional = r.deviceOptional(2);
    check_legacy_ctor_device(type, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::sparse_coo_tensor(r.tensor(0), r.tensor(1));
  } else if (r.idx == 3) {
    auto deviceOptional = r.deviceOptional(3);
    check_legacy_ctor_device(type, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::sparse_coo_tensor(r.tensor(0), r.tensor(1), r.intlist(2));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(type, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(type, deviceOptional, r.pyobject(0));
    }
    return new_with_sizes(type, r.device(1).index(), r.intlist(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

Tensor legacy_sparse_tensor_new(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, Device? device=None)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor indices, Tensor values, *, Device? device=None)",
    "new(Tensor indices, Tensor values, IntList size, *, Device? device=None)",
    "new(IntList size, *, Device? device=None)",
  });
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto deviceOptional = r.deviceOptional(0);
    check_legacy_ctor_device(type, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::empty({0}, type.options());
  } else if (r.idx == 1) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return type.unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 2) {
    // Note: this signature doesn't have a dtype, even though it has a device; it probably shouldn't
    // have a device (we should infer it).
    auto deviceOptional = r.deviceOptional(2);
    check_legacy_ctor_device(type, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::sparse_coo_tensor(r.tensor(0), r.tensor(1));
  } else if (r.idx == 3) {
    // Note: this signature doesn't have a dtype, even though it has a device; it probably shouldn't
    // have a device (we should infer it).
    auto deviceOptional = r.deviceOptional(3);
    check_legacy_ctor_device(type, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::sparse_coo_tensor(r.tensor(0), r.tensor(1), r.intlist(2));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(type, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(type, deviceOptional, r.pyobject(0));
    }
    return new_with_sizes(type, r.device(1).index(), r.intlist(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

const Type& typeWithDefault(PythonArgs& r, int64_t dtype_idx, int64_t device_idx, const Type& type) {
  const auto scalartype = r.scalartypeWithDefault(dtype_idx, type.scalarType());
  const Device types_device_type(type.device_type());
  const auto device_type = r.isNone(device_idx) ? types_device_type : r.device(device_idx).type();
  return torch::getVariableType(scalartype, *torch::getLayout(type.backend()), device_type);
}
} // namespace

Tensor legacy_tensor_ctor(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, Device? device=None)",
    "new(Storage storage)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor other)",
    "new(IntList size, *, Device? device=None)",
    "new(PyObject* data, *, Device? device=None)",
  });

  if (type.is_sparse()) {
    return legacy_sparse_tensor_ctor(type, args, kwargs);
  }

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto deviceOptional = r.deviceOptional(0);
    check_legacy_ctor_device(type, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::empty({0}, type.options());
  } else if (r.idx == 1) {
    return new_with_storage(type, r.storage(0));
  } else if (r.idx == 2) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return type.unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 3) {
    return new_with_tensor(type, r.tensor(0));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(type, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(type, deviceOptional, r.pyobject(0));
    }
    return new_with_sizes(type, r.device(1).index(), r.intlist(0));
  } else if (r.idx == 5) {
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(type, deviceOptional);
    return legacy_new_from_sequence(type, deviceOptional, r.pyobject(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

Tensor legacy_tensor_new(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, Device? device=None)",
    "new(Storage storage)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor other)",  // this doesn't have a dtype/device because it creates an alias.
    "new(IntList size, *, Device? device=None)",
    "new(PyObject* data, *, Device? device=None)",
  });

  if (type.is_sparse()) {
    return legacy_sparse_tensor_new(type, args, kwargs);
  }

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto deviceOptional = r.deviceOptional(0);
    check_legacy_ctor_device(type, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::empty({0}, type.options());
  } else if (r.idx == 1) {
    return new_with_storage(type, r.storage(0));
  } else if (r.idx == 2) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return type.unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 3) {
    return new_with_tensor(type, r.tensor(0));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(type, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(type, deviceOptional, r.pyobject(0));
    }
    return new_with_sizes(type, r.device(1).index(), r.intlist(0));
  } else if (r.idx == 5) {
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(type, deviceOptional);
    return legacy_new_from_sequence(type, r.deviceOptional(1), r.pyobject(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

Tensor legacy_new_from_data(
    const Type& type,
    c10::optional<Device> device,
    PyObject* data) {
  return internal_new_from_data(type, device, data, false, false, false);
}

Tensor sparse_coo_tensor_ctor(const Type& default_type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "sparse_coo_tensor(PyObject* indices, PyObject* values, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
    "sparse_coo_tensor(PyObject* indices, PyObject* values, IntList size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
    "sparse_coo_tensor(IntList size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    bool type_inference = r.isNone(2);
    const auto& type = typeWithDefault(r, 2, 3, default_type);
    const auto& values_type = type.toDense();
    at::DeviceGuard device_guard(r.device(3));
    // if no dtype provided, infer type based on value type.
    Tensor values = internal_new_from_data(values_type, r.deviceOptional(3), r.pyobject(1), false, true, type_inference);
    const auto& indices_type = values.type().toScalarType(kLong);
    Tensor indices = internal_new_from_data(indices_type, r.deviceOptional(3), r.pyobject(0), false, true, false);
    return at::sparse_coo_tensor(indices, values, values.options().layout(at::kSparse)).set_requires_grad(r.toBool(4));
  } else if (r.idx == 1) {
    bool type_inference = r.isNone(3);
    const auto& type = typeWithDefault(r, 3, 4, default_type);
    const auto& values_type = type.toDense();
    at::DeviceGuard device_guard(r.device(4));
    Tensor values = internal_new_from_data(values_type, r.deviceOptional(4), r.pyobject(1), false, true, type_inference);
    const auto& indices_type = values.type().toScalarType(kLong);
    Tensor indices = internal_new_from_data(indices_type, r.deviceOptional(4), r.pyobject(0), false, true, false);
    return at::sparse_coo_tensor(indices, values, r.intlist(2), values.options().layout(at::kSparse)).set_requires_grad(r.toBool(5));
  } else if (r.idx == 2) {
    const auto& type = typeWithDefault(r, 1, 2, default_type);
    at::DeviceGuard device_guard(r.device(2));
    return at::sparse_coo_tensor(r.intlist(0), type.options().layout(at::kSparse)).set_requires_grad(r.toBool(3));
  }
  throw std::runtime_error("sparse_coo_tensor(): invalid arguments");
}

Tensor tensor_ctor(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    PyObject* data = r.pyobject(0);
    if (THPVariable_Check(data)) {
      PyErr_WarnEx(PyExc_UserWarning,
        "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() "
        "or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).", 1);
    }

    bool type_inference = r.isNone(1);
    bool args_requires_grad = r.toBool(3);
    auto new_tensor = internal_new_from_data(
               typeWithDefault(r, 1, 2, type),
               r.deviceOptional(2),
               data,
               true,
               true,
               type_inference);
    new_tensor.detach_(); // ensure new_tensor a leaf node
    new_tensor.set_requires_grad(args_requires_grad);
    return new_tensor;
  }
  throw std::runtime_error("tensor(): invalid arguments");
}

Tensor as_tensor(const Type& type, PyObject* args, PyObject* kwargs) {
  // TODO: add requires_grad once we decide on semantics for sharing data.
  static PythonArgParser parser({
    "as_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    bool type_inference = r.isNone(1);
    return internal_new_from_data(
        typeWithDefault(r, 1, 2, type), r.deviceOptional(2), r.pyobject(0), false, false, type_inference);
  }
  throw std::runtime_error("tensor(): invalid arguments");
}

Tensor new_tensor(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    PyObject* data = r.pyobject(0);
    if (THPVariable_Check(data)) {
      PyErr_WarnEx(PyExc_UserWarning,
        "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() "
        "or sourceTensor.clone().detach().requires_grad_(True), rather than tensor.new_tensor(sourceTensor).", 1);
    }

    bool args_requires_grad = r.toBool(3);
    auto new_tensor = new_from_data_copy(
               typeWithDefault(r, 1, 2, type),
               r.deviceOptional(2),
               data);
    new_tensor.detach_(); // ensure new_tensor a leaf node
    new_tensor.set_requires_grad(args_requires_grad);
    return new_tensor;
  }
  throw std::runtime_error("new_tensor(): invalid arguments");
}

Tensor new_empty(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_empty(IntList size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_type = typeWithDefault(r, 1, 2, type);
    return new_with_sizes(actual_type, r.device(2).index(), r.intlist(0)).set_requires_grad(r.toBool(3));
  }
  throw std::runtime_error("new_empty(): invalid arguments");
}

Tensor new_full(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_full(IntList size, Scalar fill_value, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_type = typeWithDefault(r, 2, 3, type);
    return dispatch_full(actual_type, r.scalar(1), r.device(3).index(), r.intlist(0)).set_requires_grad(r.toBool(4));
  }
  throw std::runtime_error("new_full(): invalid arguments");
}

Tensor new_ones(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_ones(IntList size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_type = typeWithDefault(r, 1, 2, type);
    return dispatch_ones(actual_type, r.device(2).index(), r.intlist(0)).set_requires_grad(r.toBool(3));
  }
  throw std::runtime_error("new_ones(): invalid arguments");
}

Tensor new_zeros(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_zeros(IntList size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_type = typeWithDefault(r, 1, 2, type);
    return dispatch_zeros(actual_type, r.device(2).index(), r.intlist(0)).set_requires_grad(r.toBool(3));
  }
  throw std::runtime_error("new_zeros(): invalid arguments");
}

}} // namespace torch::utils
