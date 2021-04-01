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
#include <ATen/TracerMode.h>
#include <c10/core/Backend.h>
#include <c10/core/Layout.h>
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

TensorOptions build_options(c10::TensorOptions options, at::ScalarType scalar_type, const c10::optional<Device>& device=c10::nullopt) {
  options = options.dtype(scalar_type);
  if (device.has_value()) {
    return options.device(device);
  }
  return options;
}

void maybe_initialize_cuda(const Device device) {
  if (device.is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
}

// NB: It appears there is some consistency invariant between options and device, where
// if device is non-empty, its type must be consistent with the device type in
// options.
// TODO: Refactor this so we just pass everything in via options

Tensor dispatch_ones(c10::TensorOptions options, at::ScalarType scalar_type, const optional<Device>& device, IntArrayRef sizes) {
  maybe_initialize_cuda(options.device());
  pybind11::gil_scoped_release no_gil;
  return torch::ones(sizes, build_options(options, scalar_type, device));
}

Tensor new_with_sizes(c10::TensorOptions options, at::ScalarType scalar_type, const optional<Device>& device, IntArrayRef sizes) {
  maybe_initialize_cuda(options.device());
  pybind11::gil_scoped_release no_gil;
  return torch::empty(sizes, build_options(options, scalar_type, device));
}

Tensor new_with_storage(c10::TensorOptions options, at::ScalarType scalar_type, Storage storage) {
  auto tensor = at::empty({}, build_options(options, scalar_type));
  tensor.set_(std::move(storage));
  return tensor;
}

Tensor new_with_tensor(c10::TensorOptions options, at::ScalarType scalar_type, const Tensor& other) {
  options = options.dtype(scalar_type);
  TORCH_CHECK_TYPE(other.options().type_equal(options), "expected ",
                   options, " (got ", other.options(), ")");
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
  if (is_numpy_available()) {
    if (PyArray_Check(obj)) {
      return numpy_dtype_to_aten(PyArray_TYPE((PyArrayObject*)obj));
    }
    if (PyArray_CheckScalar(obj)) {
      THPObjectPtr arr(PyArray_FromScalar(obj, nullptr));
      return numpy_dtype_to_aten(PyArray_TYPE((PyArrayObject*) arr.get()));
    }
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
      if (scalarType == ScalarType::ComplexDouble) {
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
    c10::TensorOptions options,
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
    auto device = device_opt.has_value() ? *device_opt : var.device();
    pybind11::gil_scoped_release no_gil;
    maybe_initialize_cuda(device);
    return var.to(device, inferred_scalar_type, /*non_blocking=*/false, /*copy=*/copy_variables);
  }

#ifdef USE_NUMPY
  if (PyObject_HasAttrString(data, "__cuda_array_interface__")) {
    TORCH_CHECK(!pin_memory, "Can't pin tensor constructed from __cuda_array_interface__");
    auto tensor = tensor_from_cuda_array_interface(data);
    const auto& inferred_scalar_type = type_inference ? tensor.scalar_type() : scalar_type;
    auto device = device_opt.has_value() ? *device_opt : options.device();
    pybind11::gil_scoped_release no_gil;
    maybe_initialize_cuda(device);
    return tensor.to(device, inferred_scalar_type, /*non_blocking=*/false, /*copy=*/copy_numpy);
  }

  if (is_numpy_available() && PyArray_Check(data)) {
    TORCH_CHECK(!pin_memory, "Can't pin tensor constructed from numpy");
    auto tensor = tensor_from_numpy(data, /*warn_if_not_writeable=*/!copy_numpy);
    const auto& inferred_scalar_type = type_inference ? tensor.scalar_type() : scalar_type;
    auto device = device_opt.has_value() ? *device_opt : options.device();
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
    at::AutoNonVariableTypeMode guard;  // TODO: remove
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    tensor = at::empty(sizes, at::initialTensorOptions().dtype(inferred_scalar_type).pinned_memory(pin_memory));
    recursive_store(
        (char*)tensor.data_ptr(), tensor.sizes(), tensor.strides(), 0,
        inferred_scalar_type, tensor.dtype().itemsize(), data);
  }
  auto device = device_opt.has_value() ? *device_opt : options.device();
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
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    c10::optional<Device> device,
    PyObject* data) {
  return internal_new_from_data(options, scalar_type, device, data,
                                /*copy_variables=*/true, /*copy_numpy=*/true,
                                /*type_inference=*/false);
}

Tensor legacy_new_from_sequence(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    c10::optional<Device> device,
    PyObject* data) {
  if (!PySequence_Check(data)) {
    throw TypeError("new(): data must be a sequence (got %s)", Py_TYPE(data)->tp_name);
  }
  return internal_new_from_data(options, scalar_type, device, data,
                                /*copy_variables=*/false, /*copy_numpy=*/false,
                                /*type_inference=*/false);
}

// "base" here refers to the Tensor type on which the function was invoked, e.g.:
// in x.new(y), 'x' is the base.
// TODO: Rewrite this using dispatchKeyToTensorOptions
void check_base_legacy_new(c10::DispatchKey dispatch_key, at::Layout expected_layout) {
  if (expected_layout == c10::kStrided) {
    TORCH_CHECK(
        dispatch_key == c10::DispatchKey::CPU ||
            dispatch_key == c10::DispatchKey::CUDA ||
            dispatch_key == c10::DispatchKey::HIP ||
            dispatch_key == c10::DispatchKey::XLA ||
            dispatch_key == c10::DispatchKey::XPU,
        "new(): expected DispatchKey: ",
        c10::DispatchKey::CPU,
        " or ",
        c10::DispatchKey::CUDA,
        " or ",
        c10::DispatchKey::HIP,
        " or ",
        c10::DispatchKey::XLA,
        " or ",
        c10::DispatchKey::XPU,
        " but got: ",
        dispatch_key);
  } else if(expected_layout == c10::kSparse) {
    // NOTE: no sparse XLA
    TORCH_CHECK(
        dispatch_key == c10::DispatchKey::SparseCPU ||
            dispatch_key == c10::DispatchKey::SparseCUDA ||
            dispatch_key == c10::DispatchKey::SparseHIP ||
            dispatch_key == c10::DispatchKey::SparseXPU,
        "new(): expected DispatchKey: ",
        c10::DispatchKey::SparseCPU,
        " or ",
        c10::DispatchKey::SparseCUDA,
        " or ",
        c10::DispatchKey::SparseHIP,
        " or ",
        c10::DispatchKey::SparseXPU,
        " but got: ",
        dispatch_key);
  } else {
    TORCH_INTERNAL_ASSERT(false, "unexpected layout");
  }
}

// TODO: Make this accept options instead of dispatch key
void check_legacy_ctor_device(c10::DispatchKey dispatch_key, c10::optional<Device> device) {
  if (device.has_value()) {
    TORCH_CHECK(dispatchKeyToDeviceType(dispatch_key) == device.value().type(),
             "legacy constructor expects device type: ", dispatchKeyToDeviceType(dispatch_key),
             "but device type: ", device.value().type(), " was passed");
  }
}

Tensor legacy_sparse_tensor_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  auto options = dispatchKeyToTensorOptions(dispatch_key);
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
    return at::empty({0}, build_options(options, scalar_type, deviceOptional));
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
    return new_with_sizes(options, scalar_type, r.deviceOptional(1), r.intlist(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

Tensor legacy_sparse_tensor_new(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  auto options = dispatchKeyToTensorOptions(dispatch_key);
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
    return at::empty({0}, build_options(options, scalar_type));
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
    return new_with_sizes(options, scalar_type, r.deviceOptional(1), r.intlist(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

// NB: device_idx here is NOT a DeviceIndex, but index into PythonArgs
c10::TensorOptions typeIdWithDefault(PythonArgs& r, int64_t device_idx, c10::DispatchKey dispatch_key) {
  auto options = dispatchKeyToTensorOptions(dispatch_key);
  if (!r.isNone(device_idx)) {
    // TODO: This line doesn't seem to be exercised at all in tests
    options = options.device(r.device(device_idx).type());
  }
  return options;
}

} // namespace

Tensor legacy_tensor_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  auto options = dispatchKeyToTensorOptions(dispatch_key);
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

  TORCH_WARN_ONCE(
      "Legacy tensor constructor is deprecated. "
      "Use: torch.tensor(...) for creating tensors from tensor-like objects; "
      "or torch.empty(...) for creating an uninitialized tensor with specific sizes.");

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto deviceOptional = r.deviceOptional(0);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    at::OptionalDeviceGuard device_guard(deviceOptional);
    return at::empty({0}, build_options(options, scalar_type));
  } else if (r.idx == 1) {
    THPObjectPtr dtype_attr(PyObject_GetAttrString(r.pyobject(0), "dtype"));
    if (!dtype_attr) throw python_error();
    at::ScalarType storage_scalar_type = reinterpret_cast<THPDtype*>(
        dtype_attr.get())->scalar_type;
    TORCH_CHECK(
        storage_scalar_type == scalar_type,
        "Expected Storage of type ",
        scalar_type,
        " but got type ",
        storage_scalar_type,
        " for argument 1 'storage'");
    return new_with_storage(options, scalar_type, r.storage(0));
  } else if (r.idx == 2) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return at::unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 3) {
    return new_with_tensor(options, scalar_type, r.tensor(0));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(options, scalar_type, deviceOptional, r.pyobject(0));
    }
    return new_with_sizes(options, scalar_type, r.deviceOptional(1), r.intlist(0));
  } else if (r.idx == 5) {
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    return legacy_new_from_sequence(options, scalar_type, deviceOptional, r.pyobject(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

Tensor legacy_tensor_new(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  auto options = dispatchKeyToTensorOptions(dispatch_key);
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
    return at::empty({0}, build_options(options, scalar_type));
  } else if (r.idx == 1) {
    THPObjectPtr dtype_attr(PyObject_GetAttrString(r.pyobject(0), "dtype"));
    if (!dtype_attr) throw python_error();
    at::ScalarType storage_scalar_type = reinterpret_cast<THPDtype*>(
        dtype_attr.get())->scalar_type;
    TORCH_CHECK(
        storage_scalar_type == scalar_type,
        "Expected Storage of type ",
        scalar_type,
        " but got type ",
        storage_scalar_type,
        " for argument 1 'storage'");
    return new_with_storage(options, scalar_type, r.storage(0));
  } else if (r.idx == 2) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return at::unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 3) {
    return new_with_tensor(options, scalar_type, r.tensor(0));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(options, scalar_type, deviceOptional, r.pyobject(0));
    }
    return new_with_sizes(options, scalar_type, r.deviceOptional(1), r.intlist(0));
  } else if (r.idx == 5) {
    auto deviceOptional = r.deviceOptional(1);
    check_legacy_ctor_device(dispatch_key, deviceOptional);
    return legacy_new_from_sequence(options, scalar_type, r.deviceOptional(1), r.pyobject(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

Tensor indexing_tensor_from_data(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    c10::optional<Device> device,
    PyObject* data) {
  // Specific to tensor indexing, converts an indexing list to an
  // indexing tensor (type Byte or Long)
  ScalarType inferred_scalar_type = infer_scalar_type(data);
  if (inferred_scalar_type == ScalarType::Byte || inferred_scalar_type == ScalarType::Bool) {
    return internal_new_from_data(options, inferred_scalar_type, device, data,
                                  /*copy_variables=*/false, /*copy_numpy=*/false,
                                  /*type_inference=*/false);
  } else {
    return internal_new_from_data(options, scalar_type, device, data,
                                  /*copy_variables=*/false, /*copy_numpy=*/false,
                                  /*type_inference=*/false);
  }
}

// Note [Ensuring sparse values and indices match devices]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// In all places where we construct indices, we read out options from values
// (rather than use inferred_options).  Why?  This handles the case when
// values is a CUDA tensor, but indices is a non-Tensor value (and the device
// argument is not set).  Example:
//
//  torch.sparse_coo_tensor(([0, 1],), self.empty(2, 0).cuda(), (4, 0))
//
// Sparse tensors require both indices and values to live on the same device.
// If values lives on CUDA, we can infer where the indices should live, and
// should accept even ordinary index sequences (and just make sure we write them
// into the correct device).  values is the ONLY way we know that the index
// tensor should go to CUDA, so we have to get the information in somehow.
//
// This code is kind of jank.  For one, the dtype in options is silently ignored
// by internal_new_from_data.  Also, in classic janky code style, it used to
// not work quite right: if values lives on "cuda:1", before all we said was
// "this needs to be CUDA" and indices would be allocated on the wrong tensor.
// Options is more right and gets this correct.

Tensor sparse_coo_tensor_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  TORCH_INTERNAL_ASSERT(!isSparse(dispatchKeyToBackend(dispatch_key)));
  static PythonArgParser parser({
    "sparse_coo_tensor(PyObject* indices, PyObject* values, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
    "sparse_coo_tensor(PyObject* indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
    "sparse_coo_tensor(IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    bool type_inference = r.isNone(2);
    const auto inferred_options = typeIdWithDefault(r, 3, dispatch_key);
    const auto inferred_scalar_type = r.scalartypeWithDefault(2, scalar_type);
    at::OptionalDeviceGuard device_guard(r.deviceOptional(3));
    // if no dtype provided, infer type based on value type.
    Tensor values = internal_new_from_data(inferred_options, inferred_scalar_type, r.deviceOptional(3), r.pyobject(1),
                                           /*copy_variables=*/false, /*copy_numpy=*/true,
                                           /*type_inference=*/type_inference);
    // See Note [Ensuring sparse values and indices match devices]
    Tensor indices = internal_new_from_data(values.options(), kLong, r.deviceOptional(3), r.pyobject(0),
                                            /*copy_variables=*/false, /*copy_numpy=*/true,
                                            /*type_inference=*/false);
    return at::sparse_coo_tensor(indices, values, values.options().layout(at::kSparse)).set_requires_grad(r.toBool(4));
  } else if (r.idx == 1) {
    bool type_inference = r.isNone(3);
    const auto inferred_options = typeIdWithDefault(r, 4, dispatch_key);
    const auto inferred_scalar_type = r.scalartypeWithDefault(3, scalar_type);
    at::OptionalDeviceGuard device_guard(r.deviceOptional(4));
    Tensor values = internal_new_from_data(inferred_options, inferred_scalar_type, r.deviceOptional(4), r.pyobject(1),
                                           /*copy_variables=*/false, /*copy_numpy=*/true,
                                           /*type_inference=*/type_inference);
    // See Note [Ensuring sparse values and indices match devices]
    Tensor indices = internal_new_from_data(values.options(), kLong, r.deviceOptional(4), r.pyobject(0),
                                            /*copy_variables=*/false, /*copy_numpy=*/true,
                                            /*type_inference=*/false);
    return at::sparse_coo_tensor(indices, values, r.intlist(2), values.options().layout(at::kSparse)).set_requires_grad(r.toBool(5));
  } else if (r.idx == 2) {
    const auto inferred_options = typeIdWithDefault(r, 2, dispatch_key);
    const auto inferred_scalar_type = r.scalartypeWithDefault(1, scalar_type);
    at::OptionalDeviceGuard device_guard(r.deviceOptional(2));
    return at::sparse_coo_tensor(r.intlist(0), inferred_options.dtype(inferred_scalar_type).layout(at::kSparse)).set_requires_grad(r.toBool(3));
  }
  throw std::runtime_error("sparse_coo_tensor(): invalid arguments");
}

Tensor _sparse_coo_tensor_unsafe_ctor(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  TORCH_INTERNAL_ASSERT(!isSparse(dispatchKeyToBackend(dispatch_key)));
  enum {
    ARG_INDICES = 0,
    ARG_VALUES,
    ARG_SIZE,
    ARG_TYPE,
    ARG_DEVICE,
    ARG_REQUIRES_GRAD,
    ARGS_COUNT
  };
  static PythonArgParser parser({
    "_sparse_coo_tensor_unsafe(PyObject* indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });

  ParsedArgs<ARGS_COUNT> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  bool type_inference = r.isNone(ARG_TYPE);
  const auto inferred_options = typeIdWithDefault(r, ARG_DEVICE, dispatch_key);
  const auto inferred_scalar_type = r.scalartypeWithDefault(ARG_TYPE, scalar_type);
  at::OptionalDeviceGuard device_guard(r.deviceOptional(ARG_DEVICE));
  Tensor values = internal_new_from_data(inferred_options, inferred_scalar_type, r.deviceOptional(ARG_DEVICE), r.pyobject(ARG_VALUES),
                                         /*copy_variables=*/false, /*copy_numpy=*/true,
                                         /*type_inference=*/type_inference);
  // See Note [Ensuring sparse values and indices match devices]
  Tensor indices = internal_new_from_data(values.options(), kLong, r.deviceOptional(ARG_DEVICE), r.pyobject(ARG_INDICES),
                                          /*copy_variables=*/false, /*copy_numpy=*/true,
                                          /*type_inference=*/false);
  return at::_sparse_coo_tensor_unsafe(indices, values, r.intlist(ARG_SIZE), values.options().layout(at::kSparse)).set_requires_grad(r.toBool(ARG_REQUIRES_GRAD));
}

void _validate_sparse_coo_tensor_args(c10::DispatchKey dispatch_key, at::ScalarType scalar_type, PyObject* args, PyObject* kwargs) {
  auto options = dispatchKeyToTensorOptions(dispatch_key);
  static PythonArgParser parser({
    "_validate_sparse_coo_tensor(PyObject* indices, PyObject* values, IntArrayRef size)",
  });

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  Tensor values = internal_new_from_data(
      options, scalar_type, c10::nullopt, r.pyobject(1),
      /*copy_variables=*/false, /*copy_numpy=*/true, /*type_inference=*/true);
  // See Note [Ensuring sparse values and indices match devices]
  Tensor indices = internal_new_from_data(
      values.options(), kLong, c10::nullopt, r.pyobject(0),
      /*copy_variables=*/false, /*copy_numpy=*/true, /*type_inference=*/false);
  at::native::_validate_sparse_coo_tensor_args(indices, values, r.intlist(2));
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
      auto ret = PyErr_WarnEx(PyExc_UserWarning,
        "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() "
        "or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).", 1);
      if (ret != 0) throw python_error();
    }

    bool type_inference = r.isNone(1);
    bool pin_memory = r.toBool(3);
    bool args_requires_grad = r.toBool(4);
    auto new_tensor = internal_new_from_data(
               typeIdWithDefault(r, 2, dispatch_key),
               r.scalartypeWithDefault(1, scalar_type),
               r.deviceOptional(2),
               data,
               /*copy_variables=*/true,
               /*copy_numpy=*/true,
               /*type_inference=*/type_inference,
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
        /*copy_variables=*/false,
        /*copy_numpy=*/false,
        /*type_inference=*/type_inference);
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
      auto ret = PyErr_WarnEx(PyExc_UserWarning,
        "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() "
        "or sourceTensor.clone().detach().requires_grad_(True), rather than tensor.new_tensor(sourceTensor).", 1);
      if (ret != 0) throw python_error();
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
    const auto actual_options = typeIdWithDefault(r, 2, dispatch_key);
    const auto actual_scalar_type = r.scalartypeWithDefault(1, scalar_type);
    return dispatch_ones(actual_options, actual_scalar_type, r.deviceOptional(2), r.intlist(0)).set_requires_grad(r.toBool(3));
  }
  throw std::runtime_error("new_ones(): invalid arguments");
}

}} // namespace torch::utils
