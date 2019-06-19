#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/tensor_new.h>

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/numpy_stub.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_scalars.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <stdexcept>
#include <vector>

using at::Backend;
using at::Device;
using at::IntArrayRef;
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

void maybe_initialize_cuda(const TensorOptions& options) {
  if (options.device().is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
}

void maybe_initialize_cuda(const Device device) {
  if (device.is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
}

Tensor dispatch_zeros(const TensorOptions& options, optional<Device> device, IntArrayRef sizes) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::zeros(sizes, options.device(std::move(device)));
}

Tensor dispatch_ones(const TensorOptions& options, optional<Device> device, IntArrayRef sizes) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::ones(sizes, options.device(std::move(device)));
}

Tensor dispatch_full(const TensorOptions& options, Scalar fill_value, optional<Device> device, IntArrayRef sizes) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::full(sizes, fill_value, options.device(std::move(device)));
}

Tensor new_with_sizes(const TensorOptions& options, optional<Device> device, IntArrayRef sizes) {
  maybe_initialize_cuda(options);
  AutoNoGIL no_gil;
  return torch::empty(sizes, options.device(std::move(device)));
}

Tensor new_with_storage(const TensorOptions& options, Storage storage) {
  auto tensor = at::empty({}, options);
  tensor.set_(std::move(storage));
  return tensor;
}

Tensor new_with_tensor(const TensorOptions& options, const Tensor& other) {
  if (tensorTypeIdToBackend(other.type_id()) != options.backend()) {
    throw TypeError("expected %s (got %s)", toString(options.backend()), toString(tensorTypeIdToBackend(other.type_id())));
  }
  if (other.scalar_type() != at::typeMetaToScalarType(options.dtype())) {
    throw TypeError("expected %s (got %s)", toString(at::typeMetaToScalarType(options.dtype())), toString(other.scalar_type()));
  }
  return other.slice();
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
    return at::typeMetaToScalarType(torch::tensors::get_default_tensor_options().dtype());
  }
  if (THPUtils_checkLong(obj)) {
    return ScalarType::Long;
  }
  if (PyBool_Check(obj)) {
    return ScalarType::Bool;
  }
  if (THPVariable_Check(obj)) {
    auto var = reinterpret_cast<THPVariable*>(obj)->cdata;
    return var.scalar_type();
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
    if (length == 0) return at::typeMetaToScalarType(torch::tensors::get_default_tensor_options().dtype());
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

void recursive_store(char* data, IntArrayRef sizes, IntArrayRef strides, int64_t dim,
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
    const TensorOptions& options,
    c10::optional<Device> device_opt,
    PyObject* data,
    bool copy_variables,
    bool copy_numpy,
    bool type_inference,
    bool pin_memory = false) {

  if (THPUtils_checkString(data)) {
    throw TypeError("new(): invalid data type '%s'", Py_TYPE(data)->tp_name);
  }

  if (THPVariable_Check(data)) {
    TORCH_CHECK(!pin_memory, "Can't pin tensor constructed from a variable");
    auto var = reinterpret_cast<THPVariable*>(data)->cdata;
    if (copy_variables) {
      var = var.detach();
    }
    // infer the scalar type and device type; it's not expected to infer the layout since these constructors
    // are defined per-layout-type (e.g. tensor vs sparse_coo_tensor).
    const auto& inferred_scalar_type = type_inference ? var.scalar_type() : at::typeMetaToScalarType(options.dtype());
    auto device = device_opt.has_value() ? *device_opt : (type_inference ? var.device() : at::Device(options.device().type()));
    AutoNoGIL no_gil;
    maybe_initialize_cuda(device);
    return var.to(device, inferred_scalar_type, /*non_blocking=*/false, /*copy=*/copy_variables);
  }

#ifdef USE_NUMPY
  if (PyObject_HasAttrString(data, "__cuda_array_interface__")) {
    TORCH_CHECK(!pin_memory, "Can't pin tensor constructed from __cuda_array_interface__");
    auto tensor = autograd::make_variable(tensor_from_cuda_array_interface(data), /*requires_grad=*/false);
    const auto& inferred_scalar_type = type_inference ? tensor.scalar_type() : at::typeMetaToScalarType(options.dtype());
    auto device = device_opt.has_value() ? *device_opt : at::Device(options.device().type());
    AutoNoGIL no_gil;
    maybe_initialize_cuda(device);
    return tensor.to(device, inferred_scalar_type, /*non_blocking=*/false, /*copy=*/copy_numpy);
  }

  if (PyArray_Check(data)) {
    TORCH_CHECK(!pin_memory, "Can't pin tensor constructed from numpy");
    auto tensor = autograd::make_variable(tensor_from_numpy(data), /*requires_grad=*/false);
    const auto& inferred_scalar_type = type_inference ? tensor.scalar_type() : at::typeMetaToScalarType(options.dtype());
    auto device = device_opt.has_value() ? *device_opt : at::Device(options.device().type());
    AutoNoGIL no_gil;
    maybe_initialize_cuda(device);
    return tensor.to(device, inferred_scalar_type, /*non_blocking=*/false, /*copy=*/copy_numpy);
  }
#endif

  auto sizes = compute_sizes(data);
  ScalarType inferred_scalar_type = type_inference ? infer_scalar_type(data) : at::typeMetaToScalarType(options.dtype());
  auto tensor = autograd::make_variable(at::empty(sizes, at::initialTensorOptions().dtype(inferred_scalar_type).pinned_memory(pin_memory)), /*requires_grad=*/false);
  recursive_store(
      (char*)tensor.data_ptr(), tensor.sizes(), tensor.strides(), 0,
      inferred_scalar_type, tensor.dtype().itemsize(), data);
  auto device = device_opt.has_value() ? *device_opt : at::Device(options.device().type());
  AutoNoGIL no_gil;
  maybe_initialize_cuda(device);
  return tensor.to(device, inferred_scalar_type, /*non_blocking=*/false, /*copy=*/false);
}

Tensor new_from_data_copy(
    const TensorOptions& options,
    c10::optional<Device> device,
    PyObject* data) {
  return internal_new_from_data(options, std::move(device), data, true, true, false);
}

Tensor legacy_new_from_sequence(
    const TensorOptions& options,
    c10::optional<Device> device,
    PyObject* data) {
  if (!PySequence_Check(data)) {
    throw TypeError("new(): data must be a sequence (got %s)", Py_TYPE(data)->tp_name);
  }
  return internal_new_from_data(options, std::move(device), data, false, false, false);
}

void check_legacy_ctor_device(const TensorOptions& options, c10::optional<Device> device) {
  if (device.has_value()) {
    TORCH_CHECK(options.device().type() == device.value().type(),
             "legacy constructor for device type: ", options.device().type(),
             " was passed device type: ", device.value().type(),
             ", but device type must be: ", options.device().type());
  }
}

Tensor legacy_sparse_tensor_ctor(const TensorOptions& options, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, Device? device=None)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor indices, Tensor values, *, Device? device=None)",
    "new(Tensor indices, Tensor values, IntArrayRef size, *, Device? device=None)",
    "new(IntArrayRef size, *, Device? device=None)",
  });
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto deviceOptional = r.deviceOptional(0);
    check_legacy_ctor_device(options, deviceOptional);
    return at::empty({0}, options.device(r.deviceOptional(0)));
  } else if (r.idx == 1) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return autograd::make_variable(at::unsafeTensorFromTH(cdata, true));
  } else if (r.idx == 2) {
    auto deviceOptional = r.deviceOptional(2);
    check_legacy_ctor_device(options, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::sparse_coo_tensor(r.tensor(0), r.tensor(1));
  } else if (r.idx == 3) {
    auto deviceOptional = r.deviceOptional(3);
    check_legacy_ctor_device(options, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::sparse_coo_tensor(r.tensor(0), r.tensor(1), r.intlist(2));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(options, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(options, deviceOptional, r.pyobject(0));
    }
    return new_with_sizes(options, r.deviceOptional(1), r.intlist(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

Tensor legacy_sparse_tensor_new(const TensorOptions& options, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, Device? device=None)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor indices, Tensor values, *, Device? device=None)",
    "new(Tensor indices, Tensor values, IntArrayRef size, *, Device? device=None)",
    "new(IntArrayRef size, *, Device? device=None)",
  });
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto deviceOptional = r.deviceOptional(0);
    check_legacy_ctor_device(options, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::empty({0}, options);
  } else if (r.idx == 1) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return autograd::make_variable(at::unsafeTensorFromTH(cdata, true));
  } else if (r.idx == 2) {
    // Note: this signature doesn't have a dtype, even though it has a device; it probably shouldn't
    // have a device (we should infer it).
    auto deviceOptional = r.deviceOptional(2);
    check_legacy_ctor_device(options, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::sparse_coo_tensor(r.tensor(0), r.tensor(1));
  } else if (r.idx == 3) {
    // Note: this signature doesn't have a dtype, even though it has a device; it probably shouldn't
    // have a device (we should infer it).
    auto deviceOptional = r.deviceOptional(3);
    check_legacy_ctor_device(options, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::sparse_coo_tensor(r.tensor(0), r.tensor(1), r.intlist(2));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(options, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(options, deviceOptional, r.pyobject(0));
    }
    return new_with_sizes(options, r.deviceOptional(1), r.intlist(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

// NB: device_idx here is NOT a DeviceIndex, but index into PythonArgs
const TensorOptions optionsWithDefault(PythonArgs& r, int64_t dtype_idx, int64_t device_idx, const TensorOptions& options) {
  const auto scalartype = r.scalartypeWithDefault(dtype_idx, at::typeMetaToScalarType(options.dtype()));
  const auto device = r.isNone(device_idx) ? options.device() : r.device(device_idx).type();
  return options.dtype(scalartype).device(device);
}
} // namespace

Tensor legacy_tensor_ctor(const TensorOptions& options, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, Device? device=None)",
    "new(Storage storage)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor other)",
    "new(IntArrayRef size, *, Device? device=None)",
    "new(PyObject* data, *, Device? device=None)",
  });

  if (isSparse(options.backend())) {
    return legacy_sparse_tensor_ctor(options, args, kwargs);
  }

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto deviceOptional = r.deviceOptional(0);
    check_legacy_ctor_device(options, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::empty({0}, options);
  } else if (r.idx == 1) {
    return new_with_storage(options, r.storage(0));
  } else if (r.idx == 2) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return autograd::make_variable(at::unsafeTensorFromTH(cdata, true));
  } else if (r.idx == 3) {
    return new_with_tensor(options, r.tensor(0));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(options, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(options, deviceOptional, r.pyobject(0));
    }
    return new_with_sizes(options, r.deviceOptional(1), r.intlist(0));
  } else if (r.idx == 5) {
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(options, deviceOptional);
    return legacy_new_from_sequence(options, deviceOptional, r.pyobject(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

Tensor legacy_tensor_new(const TensorOptions& options, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, Device? device=None)",
    "new(Storage storage)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor other)",  // this doesn't have a dtype/device because it creates an alias.
    "new(IntArrayRef size, *, Device? device=None)",
    "new(PyObject* data, *, Device? device=None)",
  });

  if (isSparse(options.backend())) {
    return legacy_sparse_tensor_new(options, args, kwargs);
  }

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto deviceOptional = r.deviceOptional(0);
    check_legacy_ctor_device(options, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::empty({0}, options);
  } else if (r.idx == 1) {
    return new_with_storage(options, r.storage(0));
  } else if (r.idx == 2) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return autograd::make_variable(at::unsafeTensorFromTH(cdata, true));
  } else if (r.idx == 3) {
    return new_with_tensor(options, r.tensor(0));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(options, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(options, deviceOptional, r.pyobject(0));
    }
    return new_with_sizes(options, r.deviceOptional(1), r.intlist(0));
  } else if (r.idx == 5) {
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(options, deviceOptional);
    return legacy_new_from_sequence(options, r.deviceOptional(1), r.pyobject(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

Tensor indexing_tensor_from_data(
    const TensorOptions& options,
    c10::optional<Device> device,
    PyObject* data) {
  // Specific to tensor indexing, converts an indexing list to an
  // indexing tensor (type Byte or Long)
  ScalarType inferred_scalar_type = infer_scalar_type(data);
  if (inferred_scalar_type == ScalarType::Byte || inferred_scalar_type == ScalarType::Bool) {
    return internal_new_from_data(options.dtype(inferred_scalar_type), std::move(device), data, false, false, false);
  } else {
    return internal_new_from_data(options, std::move(device), data, false, false, false);
  }
}

Tensor sparse_coo_tensor_ctor(const TensorOptions& options, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "sparse_coo_tensor(PyObject* indices, PyObject* values, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
    "sparse_coo_tensor(PyObject* indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
    "sparse_coo_tensor(IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    bool type_inference = r.isNone(2);
    const auto& inferred_options = optionsWithDefault(r, 2, 3, options);
    const auto& values_options = inferred_options.layout(at::kStrided);
    at::OptionalDeviceGuard device_guard(r.deviceOptional(3));
    // if no dtype provided, infer type based on value type.
    Tensor values = internal_new_from_data(values_options, r.deviceOptional(3), r.pyobject(1), false, true, type_inference);
    const auto& indices_options = values.options().dtype(kLong);
    Tensor indices = internal_new_from_data(indices_options, r.deviceOptional(3), r.pyobject(0), false, true, false);
    return at::sparse_coo_tensor(indices, values, values.options().layout(at::kSparse)).set_requires_grad(r.toBool(4));
  } else if (r.idx == 1) {
    bool type_inference = r.isNone(3);
    const auto& inferred_options = optionsWithDefault(r, 3, 4, options);
    const auto& values_options = inferred_options.layout(at::kStrided);
    at::OptionalDeviceGuard device_guard(r.deviceOptional(4));
    Tensor values = internal_new_from_data(values_options, r.deviceOptional(4), r.pyobject(1), false, true, type_inference);
    const auto& indices_options = values.options().dtype(kLong);
    Tensor indices = internal_new_from_data(indices_options, r.deviceOptional(4), r.pyobject(0), false, true, false);
    return at::sparse_coo_tensor(indices, values, r.intlist(2), values.options().layout(at::kSparse)).set_requires_grad(r.toBool(5));
  } else if (r.idx == 2) {
    const auto& inferred_options = optionsWithDefault(r, 1, 2, options);
    at::OptionalDeviceGuard device_guard(r.deviceOptional(2));
    return at::sparse_coo_tensor(r.intlist(0), inferred_options.layout(at::kSparse)).set_requires_grad(r.toBool(3));
  }
  throw std::runtime_error("sparse_coo_tensor(): invalid arguments");
}

Tensor tensor_ctor(const TensorOptions& options, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    PyObject* data = r.pyobject(0);
    if (THPVariable_Check(data)) {
      PyErr_WarnEx(PyExc_UserWarning,
        "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() "
        "or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).", 1);
    }

    bool type_inference = r.isNone(1);
    bool pin_memory = r.toBool(3);
    bool args_requires_grad = r.toBool(4);
    auto new_tensor = internal_new_from_data(
               optionsWithDefault(r, 1, 2, options),
               r.deviceOptional(2),
               data,
               true,
               true,
               type_inference,
               pin_memory);
    new_tensor.detach_(); // ensure new_tensor a leaf node
    new_tensor.set_requires_grad(args_requires_grad);
    return new_tensor;
  }
  throw std::runtime_error("tensor(): invalid arguments");
}

Tensor as_tensor(const TensorOptions& options, PyObject* args, PyObject* kwargs) {
  // TODO: add requires_grad once we decide on semantics for sharing data.
  static PythonArgParser parser({
    "as_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    bool type_inference = r.isNone(1);
    return internal_new_from_data(
        optionsWithDefault(r, 1, 2, options),
        r.deviceOptional(2),
        r.pyobject(0),
        false,
        false,
        type_inference);
  }
  throw std::runtime_error("tensor(): invalid arguments");
}

Tensor new_tensor(const TensorOptions& options, PyObject* args, PyObject* kwargs) {
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
               optionsWithDefault(r, 1, 2, options),
               r.deviceOptional(2),
               data);
    new_tensor.detach_(); // ensure new_tensor a leaf node
    new_tensor.set_requires_grad(args_requires_grad);
    return new_tensor;
  }
  throw std::runtime_error("new_tensor(): invalid arguments");
}

Tensor new_empty(const TensorOptions& options, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_empty(IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_options = optionsWithDefault(r, 1, 2, options);
    return new_with_sizes(actual_options, r.deviceOptional(2), r.intlist(0)).set_requires_grad(r.toBool(4));
  }
  throw std::runtime_error("new_empty(): invalid arguments");
}

Tensor new_full(const TensorOptions& options, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_full(IntArrayRef size, Scalar fill_value, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_options = optionsWithDefault(r, 2, 3, options);
    return dispatch_full(actual_options, r.scalar(1), r.deviceOptional(3), r.intlist(0)).set_requires_grad(r.toBool(4));
  }
  throw std::runtime_error("new_full(): invalid arguments");
}

Tensor new_ones(const TensorOptions& options, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_ones(IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_options = optionsWithDefault(r, 1, 2, options);
    return dispatch_ones(actual_options, r.deviceOptional(2), r.intlist(0)).set_requires_grad(r.toBool(3));
  }
  throw std::runtime_error("new_ones(): invalid arguments");
}

Tensor new_zeros(const TensorOptions& options, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_zeros(IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_options = optionsWithDefault(r, 1, 2, options);
    return dispatch_zeros(actual_options, r.deviceOptional(2), r.intlist(0)).set_requires_grad(r.toBool(3));
  }
  throw std::runtime_error("new_zeros(): invalid arguments");
}

}} // namespace torch::utils
