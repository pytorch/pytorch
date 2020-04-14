#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/tensor_new.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/autograd/variable.h>
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
#include <ATen/NamedTensorUtils.h>
#include <c10/core/Backend.h>
#include <c10/core/Layout.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <stdexcept>
#include <vector>

using at::Backend;
using at::Device;
using at::DeviceType;
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

Backend backendToBackendOfDeviceType(Backend b, DeviceType d) {
  switch (d) {
    case DeviceType::CPU:
      return backendToCPU(b);
    case DeviceType::CUDA:
      return backendToCUDA(b);
    case DeviceType::HIP:
      return backendToHIP(b);
    case DeviceType::MSNPU:
      TORCH_CHECK(!isSparse(b), "Sparse not implemented for MSNPU");
      return Backend::MSNPU;
    case DeviceType::XLA:
      TORCH_CHECK(!isSparse(b), "Sparse not implemented for XLA");
      return Backend::XLA;
    default:
      AT_ERROR("Unknown device type");
  }
}

TensorOptions options(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, const c10::optional<Device>& device=c10::nullopt) {
  auto options = TensorOptions(scalar_type)
      .device(computeDeviceType(dispatch_key))
      .layout(layout_from_backend(dispatchKeyToBackend(dispatch_key)));
  if (device.has_value()) {
    return options.device(device);
  }
  return options;
}

void maybe_initialize_cuda(c10::DispatchKey dispatch_key) {
  if (backendToDeviceType(dispatchKeyToBackend(dispatch_key)) == kCUDA) {
    torch::utils::cuda_lazy_init();
  }
}

void maybe_initialize_cuda(const Device device) {
  if (device.is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
}

Tensor dispatch_zeros(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, const optional<Device>& device, IntArrayRef sizes) {
  maybe_initialize_cuda(dispatch_key);
  pybind11::gil_scoped_release no_gil;
  return torch::zeros(sizes, options(dispatch_key, scalar_type, device));
}

Tensor dispatch_ones(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, const optional<Device>& device, IntArrayRef sizes) {
  maybe_initialize_cuda(dispatch_key);
  pybind11::gil_scoped_release no_gil;
  return torch::ones(sizes, options(dispatch_key, scalar_type, device));
}

Tensor dispatch_full(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, Scalar fill_value, const optional<Device>& device, IntArrayRef sizes) {
  maybe_initialize_cuda(dispatch_key);
  pybind11::gil_scoped_release no_gil;
  return torch::full(sizes, fill_value, options(dispatch_key, scalar_type, device));
}

Tensor new_with_sizes(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, const optional<Device>& device, IntArrayRef sizes) {
  maybe_initialize_cuda(dispatch_key);
  pybind11::gil_scoped_release no_gil;
  return torch::empty(sizes, options(dispatch_key, scalar_type, device));
}

Tensor new_with_storage(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, Storage storage) {
  auto tensor = at::empty({}, options(dispatch_key, scalar_type));
  tensor.set_(std::move(storage));
  return tensor;
}

Tensor new_with_tensor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, const Tensor& other) {
  if (legacyExtractDispatchKey(other.key_set()) != dispatch_key) {
    // In temporary expression lifetime we trust
    throw TypeError("expected %s (got %s)", toString(dispatch_key), toString(other.key_set()).c_str());
  }
  if (other.scalar_type() != scalar_type) {
    throw TypeError("expected %s (got %s)", toString(scalar_type), toString(other.scalar_type()));
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
#ifdef USE_NUMPY
  if (PyArray_Check(obj)) {
    return numpy_dtype_to_aten(PyArray_TYPE((PyArrayObject*)obj));
  }
  if (PyArray_CheckScalar(obj)) {
    THPObjectPtr arr(PyArray_FromScalar(obj, nullptr));
    return numpy_dtype_to_aten(PyArray_TYPE((PyArrayObject*) arr.get()));
  }
#endif
  if (PyFloat_Check(obj)) {
    // this is always guaranteed to be a floating-point type, and makes it more
    // convenient to write e.g. torch.tensor(0.) than torch.tensor(0., dtype=torch.Tensor.dtype).
    return torch::tensors::get_default_scalar_type();
  }
  if (THPUtils_checkLong(obj)) {
    return ScalarType::Long;
  }
  if (PyBool_Check(obj)) {
    return ScalarType::Bool;
  }
  if (PyComplex_Check(obj)) {
    switch (torch::tensors::get_default_scalar_type()) {
      case ScalarType::Float: return ScalarType::ComplexFloat;
      case ScalarType::Double: return ScalarType::ComplexDouble;
      default: TORCH_CHECK(false, "invalid default scalar type for complex");
    }
  }
  if (THPVariable_Check(obj)) {
    auto var = reinterpret_cast<THPVariable*>(obj)->cdata;
    return var.scalar_type();
  }
  if (THPUtils_checkString(obj)) {
    throw TypeError("new(): invalid data type '%s'", Py_TYPE(obj)->tp_name);
  }
  if (PySequence_Check(obj)) {
    c10::optional<ScalarType> scalarType;
    auto length = PySequence_Length(obj);
    if (length < 0) throw python_error();
    // match NumPy semantics, except use default tensor type instead of double.
    if (length == 0) return torch::tensors::get_default_scalar_type();
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
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
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
    const auto& inferred_scalar_type = type_inference ? var.scalar_type() : scalar_type;
    auto device = device_opt.has_value() ? *device_opt : (type_inference ? var.device() : at::Device(computeDeviceType(dispatch_key)));
    pybind11::gil_scoped_release no_gil;
    maybe_initialize_cuda(device);
    return var.to(device, inferred_scalar_type, /*non_blocking=*/false, /*copy=*/copy_variables);
  }

#ifdef USE_NUMPY
  if (PyObject_HasAttrString(data, "__cuda_array_interface__")) {
    TORCH_CHECK(!pin_memory, "Can't pin tensor constructed from __cuda_array_interface__");
    auto tensor = tensor_from_cuda_array_interface(data);
    const auto& inferred_scalar_type = type_inference ? tensor.scalar_type() : scalar_type;
    auto device = device_opt.has_value() ? *device_opt : at::Device(computeDeviceType(dispatch_key));
    pybind11::gil_scoped_release no_gil;
    maybe_initialize_cuda(device);
    return tensor.to(device, inferred_scalar_type, /*non_blocking=*/false, /*copy=*/copy_numpy);
  }

  if (PyArray_Check(data)) {
    TORCH_CHECK(!pin_memory, "Can't pin tensor constructed from numpy");
    auto tensor = tensor_from_numpy(data);
    const auto& inferred_scalar_type = type_inference ? tensor.scalar_type() : scalar_type;
    auto device = device_opt.has_value() ? *device_opt : at::Device(computeDeviceType(dispatch_key));
    pybind11::gil_scoped_release no_gil;
    maybe_initialize_cuda(device);
    return tensor.to(device, inferred_scalar_type, /*non_blocking=*/false, /*copy=*/copy_numpy);
  }
#endif

  auto sizes = compute_sizes(data);
  ScalarType inferred_scalar_type = type_inference ? infer_scalar_type(data) : scalar_type;
  // This exists to prevent us from tracing the call to empty().  The actual
  // autograd code doesn't really matter, because requires_grad is always false
  // here.
  Tensor tensor;
  {
    at::AutoNonVariableTypeMode guard;
    tensor = at::empty(sizes, at::initialTensorOptions().dtype(inferred_scalar_type).pinned_memory(pin_memory));
    recursive_store(
        (char*)tensor.data_ptr(), tensor.sizes(), tensor.strides(), 0,
        inferred_scalar_type, tensor.dtype().itemsize(), data);
  }
  auto device = device_opt.has_value() ? *device_opt : at::Device(computeDeviceType(dispatch_key));
  pybind11::gil_scoped_release no_gil;
  maybe_initialize_cuda(device);
  // However, it is VERY important that we trace the to() call here (even
  // though the reason this is important is a hack).  Without *some* factory
  // function call that is traced at construction time, we will consider
  // a tensor constant as originating from "outside" the trace, and if you
  // try to return it directly we will fail with the error saying no
  // "no observable data dependence".  In an ideal world, we wouldn't trace
  // a to() call but I need to think harder about what exactly we should trace
  // in this case.
  return tensor.to(device, inferred_scalar_type, /*non_blocking=*/false, /*copy=*/false);
}

Tensor new_from_data_copy(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    c10::optional<Device> device,
    PyObject* data) {
  return internal_new_from_data(dispatch_key, scalar_type, std::move(device), data, true, true, false);
}

Tensor legacy_new_from_sequence(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    c10::optional<Device> device,
    PyObject* data) {
  if (!PySequence_Check(data)) {
    throw TypeError("new(): data must be a sequence (got %s)", Py_TYPE(data)->tp_name);
  }
  return internal_new_from_data(dispatch_key, scalar_type, std::move(device), data, false, false, false);
}

// "base" here refers to the Tensor type on which the function was invoked, e.g.:
// in x.new(y), 'x' is the base.
void check_base_legacy_new(c10::DispatchKey dispatch_key, at::Layout expected_layout) {
  if (expected_layout == c10::kStrided) {
    TORCH_CHECK(dispatch_key == c10::DispatchKey::CPU
                || dispatch_key == c10::DispatchKey::CUDA
                || dispatch_key == c10::DispatchKey::HIP
                || dispatch_key == c10::XLA(),
                "new(): expected DispatchKey: ", c10::DispatchKey::CPU,
                " or ", c10::DispatchKey::CUDA,
                " or ", c10::DispatchKey::HIP,
                " or ", c10::DispatchKey::XLA,
                " but got: ", dispatch_key);
  } else if(expected_layout == c10::kSparse) {
    // NOTE: no sparse XLA
    TORCH_CHECK(dispatch_key == c10::DispatchKey::SparseCPU
                || dispatch_key == c10::DispatchKey::SparseCUDA
                || dispatch_key == c10::DispatchKey::SparseHIP,
                "new(): expected DispatchKey: ", c10::DispatchKey::SparseCPU,
                " or ", c10::DispatchKey::SparseCUDA,
                " or ", c10::DispatchKey::SparseHIP,
                " but got: ", dispatch_key);
  } else {
    TORCH_INTERNAL_ASSERT(false, "unexpected layout");
  }
}

void check_legacy_ctor_device(c10::DispatchKey dispatch_key, c10::optional<Device> device) {
  if (device.has_value()) {
    TORCH_CHECK(computeDeviceType(dispatch_key) == device.value().type(),
             "legacy constructor for device type: ", computeDeviceType(dispatch_key),
             " was passed device type: ", device.value().type(),
             ", but device type must be: ", computeDeviceType(dispatch_key));
  }
}

Tensor legacy_sparse_tensor_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
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
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    return at::empty({0}, options(dispatch_key, scalar_type, deviceOptional));
  } else if (r.idx == 1) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return at::unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 2) {
    auto deviceOptional = r.deviceOptional(2);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::sparse_coo_tensor(r.tensor(0), r.tensor(1));
  } else if (r.idx == 3) {
    auto deviceOptional = r.deviceOptional(3);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::sparse_coo_tensor(r.tensor(0), r.tensor(1), r.intlist(2));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      throw TypeError("torch.SparseTensor(sequence) only accepts sizes.  Please use torch.sparse_coo_tensor() " \
                      "or construct a strided tensor and convert it to sparse via to_sparse.");
    }
    return new_with_sizes(dispatch_key, scalar_type, r.deviceOptional(1), r.intlist(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

Tensor legacy_sparse_tensor_new(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, Device? device=None)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor indices, Tensor values, *, Device? device=None)",
    "new(Tensor indices, Tensor values, IntArrayRef size, *, Device? device=None)",
    "new(IntArrayRef size, *, Device? device=None)",
  });
  check_base_legacy_new(dispatch_key, c10::kSparse);
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto deviceOptional = r.deviceOptional(0);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::empty({0}, options(dispatch_key, scalar_type));
  } else if (r.idx == 1) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return at::unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 2) {
    // Note: this signature doesn't have a dtype, even though it has a device; it probably shouldn't
    // have a device (we should infer it).
    auto deviceOptional = r.deviceOptional(2);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::sparse_coo_tensor(r.tensor(0), r.tensor(1));
  } else if (r.idx == 3) {
    // Note: this signature doesn't have a dtype, even though it has a device; it probably shouldn't
    // have a device (we should infer it).
    auto deviceOptional = r.deviceOptional(3);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::sparse_coo_tensor(r.tensor(0), r.tensor(1), r.intlist(2));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      throw TypeError("SparseTensor.new(sequence) only accepts sizes.  Please use torch.sparse_coo_tensor() " \
                      "or construct a strided tensor and convert it to sparse via to_sparse.");
    }
    return new_with_sizes(dispatch_key, scalar_type, r.deviceOptional(1), r.intlist(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

// NB: device_idx here is NOT a DeviceIndex, but index into PythonArgs
c10::DispatchKey typeIdWithDefault(PythonArgs& r, int64_t device_idx, c10::DispatchKey dispatch_key) {
  auto device_type = r.isNone(device_idx) ? computeDeviceType(dispatch_key) : r.device(device_idx).type();
  return backendToDispatchKey(backendToBackendOfDeviceType(dispatchKeyToBackend(dispatch_key), device_type));
}

// NB: device_idx here is NOT a DeviceIndex, but index into PythonArgs
c10::DispatchKey denseTypeIdWithDefault(PythonArgs& r, int64_t device_idx, c10::DispatchKey dispatch_key) {
  auto device_type = r.isNone(device_idx) ? computeDeviceType(dispatch_key) : r.device(device_idx).type();
  return backendToDispatchKey(toDense(backendToBackendOfDeviceType(dispatchKeyToBackend(dispatch_key), device_type)));
}
} // namespace

Tensor legacy_tensor_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, Device? device=None)",
    "new(Storage storage)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor other)",
    "new(IntArrayRef size, *, Device? device=None)",
    "new(PyObject* data, *, Device? device=None)",
  });

  if (isSparse(dispatchKeyToBackend(dispatch_key))) {
    return legacy_sparse_tensor_ctor(dispatch_key, scalar_type, args, kwargs);
  }

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto deviceOptional = r.deviceOptional(0);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::empty({0}, options(dispatch_key, scalar_type));
  } else if (r.idx == 1) {
    return new_with_storage(dispatch_key, scalar_type, r.storage(0));
  } else if (r.idx == 2) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return at::unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 3) {
    return new_with_tensor(dispatch_key, scalar_type, r.tensor(0));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(dispatch_key, scalar_type, deviceOptional, r.pyobject(0));
    }
    return new_with_sizes(dispatch_key, scalar_type, r.deviceOptional(1), r.intlist(0));
  } else if (r.idx == 5) {
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    return legacy_new_from_sequence(dispatch_key, scalar_type, deviceOptional, r.pyobject(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

Tensor legacy_tensor_new(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, Device? device=None)",
    "new(Storage storage)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor other)",  // this doesn't have a dtype/device because it creates an alias.
    "new(IntArrayRef size, *, Device? device=None)",
    "new(PyObject* data, *, Device? device=None)",
  });

  if (isSparse(dispatchKeyToBackend(dispatch_key))) {
    return legacy_sparse_tensor_new(dispatch_key, scalar_type, args, kwargs);
  }

  check_base_legacy_new(dispatch_key, c10::kStrided);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto deviceOptional = r.deviceOptional(0);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::empty({0}, options(dispatch_key, scalar_type));
  } else if (r.idx == 1) {
    return new_with_storage(dispatch_key, scalar_type, r.storage(0));
  } else if (r.idx == 2) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return at::unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 3) {
    return new_with_tensor(dispatch_key, scalar_type, r.tensor(0));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(dispatch_key, scalar_type, deviceOptional, r.pyobject(0));
    }
    return new_with_sizes(dispatch_key, scalar_type, r.deviceOptional(1), r.intlist(0));
  } else if (r.idx == 5) {
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    return legacy_new_from_sequence(dispatch_key, scalar_type, r.deviceOptional(1), r.pyobject(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

Tensor indexing_tensor_from_data(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    c10::optional<Device> device,
    PyObject* data) {
  // Specific to tensor indexing, converts an indexing list to an
  // indexing tensor (type Byte or Long)
  ScalarType inferred_scalar_type = infer_scalar_type(data);
  if (inferred_scalar_type == ScalarType::Byte || inferred_scalar_type == ScalarType::Bool) {
    return internal_new_from_data(dispatch_key, inferred_scalar_type, std::move(device), data, false, false, false);
  } else {
    return internal_new_from_data(dispatch_key, scalar_type, std::move(device), data, false, false, false);
  }
}

Tensor sparse_coo_tensor_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "sparse_coo_tensor(PyObject* indices, PyObject* values, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
    "sparse_coo_tensor(PyObject* indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
    "sparse_coo_tensor(IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    bool type_inference = r.isNone(2);
    const auto inferred_dispatch_key = denseTypeIdWithDefault(r, 3, dispatch_key);
    const auto inferred_scalar_type = r.scalartypeWithDefault(2, scalar_type);
    at::OptionalDeviceGuard device_guard(r.deviceOptional(3));
    // if no dtype provided, infer type based on value type.
    Tensor values = internal_new_from_data(inferred_dispatch_key, inferred_scalar_type, r.deviceOptional(3), r.pyobject(1), false, true, type_inference);
    Tensor indices = internal_new_from_data(legacyExtractDispatchKey(values.key_set()), kLong, r.deviceOptional(3), r.pyobject(0), false, true, false);
    return at::sparse_coo_tensor(indices, values, values.options().layout(at::kSparse)).set_requires_grad(r.toBool(4));
  } else if (r.idx == 1) {
    bool type_inference = r.isNone(3);
    const auto inferred_dispatch_key = denseTypeIdWithDefault(r, 4, dispatch_key);
    const auto inferred_scalar_type = r.scalartypeWithDefault(3, scalar_type);
    at::OptionalDeviceGuard device_guard(r.deviceOptional(4));
    Tensor values = internal_new_from_data(inferred_dispatch_key, inferred_scalar_type, r.deviceOptional(4), r.pyobject(1), false, true, type_inference);
    Tensor indices = internal_new_from_data(legacyExtractDispatchKey(values.key_set()), kLong, r.deviceOptional(4), r.pyobject(0), false, true, false);
    return at::sparse_coo_tensor(indices, values, r.intlist(2), values.options().layout(at::kSparse)).set_requires_grad(r.toBool(5));
  } else if (r.idx == 2) {
    const auto inferred_dispatch_key = typeIdWithDefault(r, 2, dispatch_key);
    const auto inferred_scalar_type = r.scalartypeWithDefault(1, scalar_type);
    at::OptionalDeviceGuard device_guard(r.deviceOptional(2));
    return at::sparse_coo_tensor(r.intlist(0), options(inferred_dispatch_key, inferred_scalar_type).layout(at::kSparse)).set_requires_grad(r.toBool(3));
  }
  throw std::runtime_error("sparse_coo_tensor(): invalid arguments");
}

Tensor tensor_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, DimnameList? names=None)",
  });

  constexpr int ctor_num_args = 6;
  ParsedArgs<ctor_num_args> parsed_args;
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
               typeIdWithDefault(r, 2, dispatch_key),
               r.scalartypeWithDefault(1, scalar_type),
               r.deviceOptional(2),
               data,
               true,
               true,
               type_inference,
               pin_memory);
    auto names = r.toDimnameListOptional(5);
    if (names) {
      at::namedinference::propagate_names(new_tensor, *names, /*validate_names=*/true);
    }
    new_tensor.detach_(); // ensure new_tensor a leaf node
    new_tensor.set_requires_grad(args_requires_grad);
    return new_tensor;
  }
  throw std::runtime_error("tensor(): invalid arguments");
}

Tensor as_tensor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  // TODO: add requires_grad once we decide on semantics for sharing data.
  static PythonArgParser parser({
    "as_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    bool type_inference = r.isNone(1);
    return internal_new_from_data(
        typeIdWithDefault(r, 2, dispatch_key),
        r.scalartypeWithDefault(1, scalar_type),
        r.deviceOptional(2),
        r.pyobject(0),
        false,
        false,
        type_inference);
  }
  throw std::runtime_error("tensor(): invalid arguments");
}

Tensor new_tensor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
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
               typeIdWithDefault(r, 2, dispatch_key),
               r.scalartypeWithDefault(1, scalar_type),
               r.deviceOptional(2),
               data);
    new_tensor.detach_(); // ensure new_tensor a leaf node
    new_tensor.set_requires_grad(args_requires_grad);
    return new_tensor;
  }
  throw std::runtime_error("new_tensor(): invalid arguments");
}

Tensor new_ones(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_ones(IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto actual_dispatch_key = typeIdWithDefault(r, 2, dispatch_key);
    const auto actual_scalar_type = r.scalartypeWithDefault(1, scalar_type);
    return dispatch_ones(actual_dispatch_key, actual_scalar_type, r.deviceOptional(2), r.intlist(0)).set_requires_grad(r.toBool(3));
  }
  throw std::runtime_error("new_ones(): invalid arguments");
}

}} // namespace torch::utils
