#include <Python.h>
#include "tensor_new.h"

#include <ATen/ATen.h>
#include <ATen/Error.h>
#include <ATen/optional.h>

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Size.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/auto_gpu.h"
#include "torch/csrc/utils/cuda_lazy_init.h"
#include "torch/csrc/utils/numpy_stub.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/python_scalars.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/tensor_conversion_dispatch.h"
#include "torch/csrc/utils/tensor_numpy.h"

static const int MAX_DIMS = 128;

using namespace at;

namespace torch { namespace utils {

static void maybe_initialize_cuda(const at::Type &type) {
  if (type.is_cuda()) {
    torch::utils::cuda_lazy_init();
  }
}

static Tensor dispatch_zeros(const Type& type, int device, IntList sizes) {
  maybe_initialize_cuda(type);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return type.zeros(sizes);
}

static Tensor dispatch_ones(const Type& type, int device, IntList sizes) {
  maybe_initialize_cuda(type);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return type.ones(sizes);
}

static Tensor dispatch_full(const Type& type, Scalar fill_value, int device, IntList sizes) {
  maybe_initialize_cuda(type);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return type.full(sizes, fill_value);
}

static Tensor new_with_sizes(const Type& type, int device, IntList sizes) {
  maybe_initialize_cuda(type);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return type.tensor(sizes);
}

static Tensor new_with_storage(const Type& type, Storage& storage) {
  auto tensor = type.tensor();
  tensor.set_(storage);
  return tensor;
}

static Tensor new_with_tensor(const Type& type, Tensor other) {
  if (other.type() != type) {
    throw TypeError("expected %s (got %s)", type.toString(), other.type().toString());
  }
  return other.slice();
}

static Tensor new_with_type_conversion(const Type& type, Tensor other, int64_t device) {
  return dispatch_type_conversion(other, type, device, false);
}

static Tensor new_with_tensor_copy(const Type& type, Tensor other, int64_t device) {
  maybe_initialize_cuda(type);
  AutoNoGIL no_gil;
  AutoGPU auto_gpu(device);
  return type.copy(other);
}

static std::vector<int64_t> compute_sizes(PyObject* seq) {
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
    seq = handle.get();
  }

  return sizes;
}

static ScalarType infer_scalar_type(PyObject *obj) {
  if (PyFloat_Check(obj)) {
    // this is always guaranteed to be a floating-point type, and makes it more
    // convenient to write e.g. torch.tensor(0.) than torch.tensor(0., dtype=torch.Tensor.dtype).
    return torch::tensor::get_default_tensor_type().scalarType();
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
#ifdef WITH_NUMPY
  if (PyArray_Check(obj)) {
    auto array = (PyArrayObject*)obj;
    return numpy_dtype_to_aten(PyArray_TYPE(array));
  }
#endif
  if (PySequence_Check(obj)) {
    at::optional<ScalarType> scalarType;
    auto length = PySequence_Length(obj);
    if (length < 0) throw python_error();
    // match NumPy semantics, except use default tensor type instead of double.
    if (length == 0) return torch::tensor::get_default_tensor_type().scalarType();
    for (int i = 0; i < length; ++i) {
      THPObjectPtr handle(PySequence_GetItem(obj, i));
      if (!handle) throw python_error();
      ScalarType item_scalarType = infer_scalar_type(handle.get());
      scalarType = (scalarType) ?
          at::promoteTypes(*scalarType, item_scalarType) : item_scalarType;
      if (scalarType == ScalarType::Double) {
        // this won't change (unless we hit undefined, but that will fail later).
        return *scalarType;
      }
    }
    return *scalarType;
  }
  AT_ERROR("Could not infer dtype of %s", Py_TYPE(obj)->tp_name);
}

static void recursive_store(char* data, IntList sizes, IntList strides, int64_t dim,
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

static Tensor internal_new_from_data(const Type & type, int device, PyObject* data,
                                     bool copy_variables, bool copy_numpy,
                                     bool type_inference) {
  if (THPUtils_checkString(data)) {
    throw TypeError("new(): invalid data type '%s'", Py_TYPE(data)->tp_name);
  }

  if (THPVariable_Check(data)) {
      auto var = reinterpret_cast<THPVariable*>(data)->cdata;
      const auto& type_to_use = type_inference ? var.type() : type;
      return copy_variables ? new_with_tensor_copy(type_to_use, var, device) :
                              new_with_type_conversion(type_to_use, var, device);
  }

#ifdef WITH_NUMPY
  if (PyArray_Check(data)) {
    auto tensor = autograd::make_variable(tensor_from_numpy(data), /*requires_grad=*/false);
    const auto& type_to_use = type_inference ? tensor.type() : type;
    return copy_numpy ? new_with_tensor_copy(type_to_use, tensor, device) :
                        new_with_type_conversion(type_to_use, tensor, device);
  }
#endif

  auto sizes = compute_sizes(data);
  ScalarType scalarType = type_inference ? infer_scalar_type(data) : type.scalarType();
  auto tensor = autograd::make_variable(CPU(scalarType).tensor(sizes), /*requires_grad=*/false);
  recursive_store(
      (char*)tensor.data_ptr(), tensor.sizes(), tensor.strides(), 0,
      scalarType, tensor.type().elementSizeInBytes(), data);
  const auto& type_to_use = type_inference ? type.toScalarType(scalarType) : type;
  return new_with_type_conversion(type_to_use, tensor, device);
}

Tensor legacy_new_from_data(const Type & type, int device, PyObject *data) {
  return internal_new_from_data(type, device, data, false, false, false);
}

static Tensor new_from_data_copy(const Type & type, int device, PyObject *data) {
  return internal_new_from_data(type, device, data, true, true, false);
}

static Tensor legacy_new_from_sequence(const Type & type, int device, PyObject* data) {
  if (!PySequence_Check(data)) {
    throw TypeError("new(): data must be a sequence (got %s)", Py_TYPE(data)->tp_name);
  }
  return legacy_new_from_data(type, device, data);
}

static Tensor legacy_sparse_tensor_ctor(const Type& type, PyObject* args, PyObject* kwargs) {
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
    AutoGPU auto_gpu(r.deviceInt64(0));
    return type.tensor();
  } else if (r.idx == 1) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return type.unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 2) {
    AutoGPU auto_gpu(r.deviceInt64(2));
    return type.sparse_coo_tensor(r.tensor(0), r.tensor(1));
  } else if (r.idx == 3) {
    AutoGPU auto_gpu(r.deviceInt64(3));
    return type.sparse_coo_tensor(r.tensor(0), r.tensor(1), r.intlist(2));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(type, r.deviceInt64(1), r.pyobject(0));
    }
    return new_with_sizes(type, r.deviceInt64(1), r.intlist(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

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
    AutoGPU auto_gpu(r.deviceInt64(0));
    return type.tensor();
  } else if (r.idx == 1) {
    return new_with_storage(type, *r.storage(0));
  } else if (r.idx == 2) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return type.unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 3) {
    return new_with_tensor(type, r.tensor(0));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(type, r.deviceInt64(1), r.pyobject(0));
    }
    return new_with_sizes(type, r.deviceInt64(1), r.intlist(0));
  } else if (r.idx == 5) {
    return legacy_new_from_sequence(type, r.deviceInt64(1), r.pyobject(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

static Tensor legacy_sparse_tensor_new(const Type& type, PyObject* args, PyObject* kwargs) {
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
    AutoGPU auto_gpu(r.deviceInt64(0));
    return type.tensor();
  } else if (r.idx == 1) {
    auto cdata = reinterpret_cast<void*>(r.deviceInt64(0));
    return type.unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 2) {
    // Note: this signature doesn't have a dtype, even though it has a device; it probably shouldn't
    // have a device (we should infer it).
    AutoGPU auto_gpu(r.deviceInt64(2));
    return type.sparse_coo_tensor(r.tensor(0), r.tensor(1));
  } else if (r.idx == 3) {
    // Note: this signature doesn't have a dtype, even though it has a device; it probably shouldn't
    // have a device (we should infer it).
    AutoGPU auto_gpu(r.deviceInt64(3));
    return type.sparse_coo_tensor(r.tensor(0), r.tensor(1), r.intlist(2));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(type, r.deviceInt64(1), r.pyobject(0));
    }
    return new_with_sizes(type, r.deviceInt64(1), r.intlist(0));
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
    AutoGPU auto_gpu(r.deviceInt64(0));
    return type.tensor();
  } else if (r.idx == 1) {
    return new_with_storage(type, *r.storage(0));
  } else if (r.idx == 2) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return type.unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 3) {
    return new_with_tensor(type, r.tensor(0));
  } else if (r.idx == 4) {
    PyObject* arg = r.pyobject(0);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(type, r.deviceInt64(1), r.pyobject(0));
    }
    return new_with_sizes(type, r.deviceInt64(1), r.intlist(0));
  } else if (r.idx == 5) {
    return legacy_new_from_sequence(type, r.deviceInt64(1), r.pyobject(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

static const Type& typeWithDefault(PythonArgs& r, int64_t dtype_idx, int64_t device_idx, const Type& type) {
  auto scalartype = r.scalartypeWithDefault(dtype_idx, type.scalarType());
  auto types_device_type = torch::getDeviceType(type);
  auto device_type = r.isNone(device_idx) ? types_device_type : r.device(device_idx).type;
  return torch::getType(scalartype, *torch::getLayout(type.backend()), device_type);
}

static Tensor set_requires_grad(Tensor self, bool requires_grad) {
  static_cast<torch::autograd::Variable&>(self).set_requires_grad(requires_grad);
  return self;
}

Tensor sparse_coo_tensor_ctor(const Type& type, PyObject* args, PyObject* kwargs) {
  Backend sparse_backend = type.is_cuda() ? kSparseCUDA : kSparseCPU;
  const auto& default_sparse_type = type.toBackend(sparse_backend);

  static PythonArgParser parser({
    "sparse_coo_tensor(PyObject* indices, PyObject* values, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
    "sparse_coo_tensor(PyObject* indices, PyObject* values, IntList size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    bool type_inference = r.isNone(2);
    const auto& sparse_type = typeWithDefault(r, 2, 3, default_sparse_type);
    const auto& dense_type = sparse_type.toBackend(sparse_type.is_cuda() ? kCUDA : kCPU);
    const auto& index_type = dense_type.toScalarType(kLong);
    AutoGPU autogpu(r.deviceInt64(3));
    // explanation of booleans: allow variables, do type conversion of them, copy numpy data
    Tensor indices = internal_new_from_data(index_type, -1, r.pyobject(0), false, true, false);
    Tensor values = internal_new_from_data(dense_type, -1, r.pyobject(1), false, true, type_inference);
    const auto& sparse_type_to_use = values.type().toBackend(values.type().is_cuda() ? kSparseCUDA : kSparseCPU);
    return set_requires_grad(sparse_type_to_use.sparse_coo_tensor(indices, values), r.toBool(4));
  } else if (r.idx == 1) {
    bool type_inference = r.isNone(3);
    const auto& sparse_type = typeWithDefault(r, 3, 4, default_sparse_type);
    const auto& dense_type = sparse_type.toBackend(sparse_type.is_cuda() ? kCUDA : kCPU);
    const auto& index_type = dense_type.toScalarType(kLong);
    AutoGPU autogpu(r.deviceInt64(4));
    // explanation of booleans: allow variables, do type conversion of them, copy numpy data
    Tensor indices = internal_new_from_data(index_type, -1, r.pyobject(0), false, true, false);
    Tensor values = internal_new_from_data(dense_type, -1, r.pyobject(1), false, true, type_inference);
    const auto& sparse_type_to_use = values.type().toBackend(values.type().is_cuda() ? kSparseCUDA : kSparseCPU);
    return set_requires_grad(sparse_type_to_use.sparse_coo_tensor(indices, values, r.intlist(2)), r.toBool(5));
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
    bool type_inference = r.isNone(1);
    return set_requires_grad(internal_new_from_data(
        typeWithDefault(r, 1, 2, type), r.deviceInt64(2), r.pyobject(0), true, true, type_inference), r.toBool(3));
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
    return set_requires_grad(new_from_data_copy(
        typeWithDefault(r, 1, 2, type), r.deviceInt64(2), r.pyobject(0)), r.toBool(3));
  }
  throw std::runtime_error("new_tensor(): invalid arguments");
}

Tensor new_empty(const at::Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_empty(IntList size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_type = typeWithDefault(r, 1, 2, type);
    return set_requires_grad(new_with_sizes(actual_type, r.deviceInt64(2), r.intlist(0)), r.toBool(3));
  }
  throw std::runtime_error("new_empty(): invalid arguments");
}

Tensor new_full(const at::Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_full(IntList size, Scalar fill_value, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_type = typeWithDefault(r, 2, 3, type);
    return set_requires_grad(dispatch_full(actual_type, r.scalar(1), r.deviceInt64(3), r.intlist(0)), r.toBool(4));
  }
  throw std::runtime_error("new_full(): invalid arguments");
}

Tensor new_ones(const at::Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_ones(IntList size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_type = typeWithDefault(r, 1, 2, type);
    return set_requires_grad(dispatch_ones(actual_type, r.deviceInt64(2), r.intlist(0)), r.toBool(3));
  }
  throw std::runtime_error("new_ones(): invalid arguments");
}

Tensor new_zeros(const at::Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_zeros(IntList size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_type = typeWithDefault(r, 1, 2, type);
    return set_requires_grad(dispatch_zeros(actual_type, r.deviceInt64(2), r.intlist(0)), r.toBool(3));
  }
  throw std::runtime_error("new_zeros(): invalid arguments");
}

}} // namespace torch::utils
