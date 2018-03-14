#include <Python.h>
#include "tensor_new.h"

#include <ATen/ATen.h>

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
  if (other.type() != type) {
    maybe_initialize_cuda(type);
    AutoNoGIL no_gil;
    AutoGPU auto_gpu(device);
    other = other.toType(type);
  }
  return other;
}

static Tensor new_with_tensor_copy(const Type& type, Tensor other, int64_t device) {
  AutoGPU auto_gpu(device);
  AutoNoGIL no_gil;
  maybe_initialize_cuda(type);
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

static Tensor internal_new_from_data(const Type & type, int device, PyObject* data, bool always_copy) {
  if (THPUtils_checkString(data)) {
    throw TypeError("new(): invalid data type '%s'", Py_TYPE(data)->tp_name);
  }
#ifdef WITH_NUMPY
  if (PyArray_Check(data)) {
    auto tensor = autograd::make_variable(tensor_from_numpy(data), /*requires_grad=*/false);
    return always_copy ? new_with_tensor_copy(type, tensor, device) : new_with_type_conversion(type, tensor, device);
  }
#endif

  auto sizes = compute_sizes(data);
  auto tensor = autograd::make_variable(CPU(type.scalarType()).tensor(sizes), /*requires_grad=*/false);
  recursive_store(
      (char*)tensor.data_ptr(), tensor.sizes(), tensor.strides(), 0,
      type.scalarType(), tensor.type().elementSizeInBytes(), data);
  return new_with_type_conversion(type, tensor, device);
}

Tensor legacy_new_from_data(const Type & type, int device, PyObject *data) {
  return internal_new_from_data(type, device, data, false);
}

static Tensor new_from_data_copy(const Type & type, int device, PyObject *data) {
  return internal_new_from_data(type, device, data, true);
}

static Tensor legacy_new_from_sequence(const Type & type, int device, PyObject* data) {
  if (!PySequence_Check(data)) {
    throw TypeError("new(): data must be a sequence (got %s)", Py_TYPE(data)->tp_name);
  }
  return legacy_new_from_data(type, device, data);
}

static void check_is_dense(const Type& type) {
  if (type.is_sparse()) {
    std::ostringstream oss;
    oss << "new(..) on a dense tensor can only be called with a dense dtype, got: ";
    oss << torch::getDtype(type)->name;
    throw TypeError(oss.str().c_str());
  }
}

static void check_is_sparse(const Type& type) {
  if (!type.is_sparse()) {
    std::ostringstream oss;
    oss << "new(..) on a spase tensor can only be called with a sparse dtype, got: ";
    oss << torch::getDtype(type)->name;
    throw TypeError(oss.str().c_str());
  }
}

static Tensor legacy_sparse_tensor_ctor(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, int64_t? device=-1)",
    "new(IntList size, *, int64_t? device=-1)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor indices, Tensor values, *, int64_t? device=-1)",
    "new(Tensor indices, Tensor values, IntList size, *, int64_t? device=-1)",
  });
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    AutoGPU auto_gpu(r.toInt64(0));
    return type.tensor();
  } else if (r.idx == 1) {
    PyObject* arg = r.pyobject(0);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(type, r.toInt64(1), r.pyobject(0));
    }
    return new_with_sizes(type, r.toInt64(1), r.intlist(0));
  } else if (r.idx == 2) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return type.unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 3) {
    AutoGPU auto_gpu(r.toInt64(2));
    return type.sparse_coo_tensor(r.tensor(0), r.tensor(1));
  } else if (r.idx == 4) {
    AutoGPU auto_gpu(r.toInt64(3));
    return type.sparse_coo_tensor(r.tensor(0), r.tensor(1), r.intlist(2));
  }
  throw std::runtime_error("new(): invalid arguments");
}

Tensor legacy_tensor_ctor(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, int64_t? device=-1)",
    "new(IntList size, *, int64_t? device=-1)",
    "new(Storage storage)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor other)",
    "new(PyObject* data, *, int64_t? device=-1)",
  });

  if (type.is_sparse()) {
    return legacy_sparse_tensor_ctor(type, args, kwargs);
  }

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    AutoGPU auto_gpu(r.toInt64(0));
    return type.tensor();
  } else if (r.idx == 1) {
    PyObject* arg = r.pyobject(0);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(type, r.toInt64(1), r.pyobject(0));
    }
    return new_with_sizes(type, r.toInt64(1), r.intlist(0));
  } else if (r.idx == 2) {
    return new_with_storage(type, *r.storage(0));
  } else if (r.idx == 3) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return type.unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 4) {
    return new_with_tensor(type, r.tensor(0));
  } else if (r.idx == 5) {
    return legacy_new_from_sequence(type, r.toInt64(1), r.pyobject(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

static Tensor legacy_sparse_tensor_new(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, Type dtype=None, int64_t? device=-1)",
    "new(IntList size, *, Type dtype=None, int64_t? device=-1)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor indices, Tensor values, *, int64_t? device=-1)",
    "new(Tensor indices, Tensor values, IntList size, *, int64_t? device=-1)",
  });
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_type = r.typeWithDefault(0, type);
    check_is_sparse(actual_type);
    maybe_initialize_cuda(actual_type);
    AutoGPU auto_gpu(r.toInt64(1));
    return actual_type.tensor();
  } else if (r.idx == 1) {
    PyObject* arg = r.pyobject(0);
    const auto& actual_type = r.typeWithDefault(1, type);
    check_is_sparse(actual_type);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(actual_type, r.toInt64(2), r.pyobject(0));
    }
    return new_with_sizes(actual_type, r.toInt64(2), r.intlist(0));
  } else if (r.idx == 2) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return type.unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 3) {
    // Note: this signature doesn't have a dtype, even though it has a device; it probably shouldn't
    // have a device (we should infer it).
    AutoGPU auto_gpu(r.toInt64(2));
    return type.sparse_coo_tensor(r.tensor(0), r.tensor(1));
  } else if (r.idx == 4) {
    // Note: this signature doesn't have a dtype, even though it has a device; it probably shouldn't
    // have a device (we should infer it).
    AutoGPU auto_gpu(r.toInt64(3));
    return type.sparse_coo_tensor(r.tensor(0), r.tensor(1), r.intlist(2));
  }
  throw std::runtime_error("new(): invalid arguments");
}

Tensor legacy_tensor_new(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, Type dtype=None, int64_t? device=-1)",
    "new(IntList size, *, Type dtype=None, int64_t? device=-1)",
    "new(Storage storage)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor other)",  // this doesn't have a dtype/device because it creates an alias.
    "new(PyObject* data, *, Type dtype=None, int64_t? device=-1)",
  });

  if (type.is_sparse()) {
    return legacy_sparse_tensor_new(type, args, kwargs);
  }

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_type = r.typeWithDefault(0, type);
    check_is_dense(actual_type);
    maybe_initialize_cuda(actual_type);
    AutoGPU auto_gpu(r.toInt64(1));
    return actual_type.tensor();
  } else if (r.idx == 1) {
    PyObject* arg = r.pyobject(0);
    const auto& actual_type = r.typeWithDefault(1, type);
    check_is_dense(actual_type);
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return legacy_new_from_sequence(actual_type, r.toInt64(2), r.pyobject(0));
    }
    return new_with_sizes(actual_type, r.toInt64(2), r.intlist(0));
  } else if (r.idx == 2) {
    return new_with_storage(type, *r.storage(0));
  } else if (r.idx == 3) {
    auto cdata = reinterpret_cast<void*>(r.toInt64(0));
    return type.unsafeTensorFromTH(cdata, true);
  } else if (r.idx == 4) {
    return new_with_tensor(type, r.tensor(0));
  } else if (r.idx == 5) {
    const auto& actual_type = r.typeWithDefault(1, type);
    check_is_dense(actual_type);
    return legacy_new_from_sequence(actual_type, r.toInt64(2), r.pyobject(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

static Tensor set_requires_grad(Tensor self, bool requires_grad) {
  static_cast<torch::autograd::Variable&>(self).set_requires_grad(requires_grad);
  return self;
}

Tensor new_tensor(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_tensor(Tensor other, *, Type dtype=None, int64_t? device=-1, bool requires_grad=False)",
    "new_tensor(PyObject* data, *, Type dtype=None, int64_t? device=-1, bool requires_grad=False)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return set_requires_grad(new_with_tensor_copy(r.typeWithDefault(1, type), r.tensor(0), r.toInt64(2)), r.toBool(3));
  } else if (r.idx == 1) {
    return set_requires_grad(new_from_data_copy(r.typeWithDefault(1, type), r.toInt64(2), r.pyobject(0)), r.toBool(3));
  }
  throw std::runtime_error("new_tensor(): invalid arguments");
}

Tensor new_empty(const at::Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_empty(IntList size, *, Type dtype=None, int64_t? device=-1, bool requires_grad=False)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_type = r.typeWithDefault(1, type);
    return set_requires_grad(new_with_sizes(actual_type, r.toInt64(2), r.intlist(0)), r.toBool(3));
  }
  throw std::runtime_error("new_empty(): invalid arguments");
}

Tensor new_full(const at::Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_full(IntList size, Scalar fill_value, *, Type dtype=None, int64_t? device=-1, bool requires_grad=False)",
  });

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_type = r.typeWithDefault(2, type);
    return set_requires_grad(dispatch_full(actual_type, r.scalar(1), r.toInt64(3), r.intlist(0)), r.toBool(4));
  }
  throw std::runtime_error("new_full(): invalid arguments");
}

Tensor new_ones(const at::Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_ones(IntList size, *, Type dtype=None, int64_t? device=-1, bool requires_grad=False)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_type = r.typeWithDefault(1, type);
    return set_requires_grad(dispatch_ones(actual_type, r.toInt64(2), r.intlist(0)), r.toBool(3));
  }
  throw std::runtime_error("new_ones(): invalid arguments");
}

Tensor new_zeros(const at::Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new_zeros(IntList size, *, Type dtype=None, int64_t? device=-1, bool requires_grad=False)",
  });

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    const auto& actual_type = r.typeWithDefault(1, type);
    return set_requires_grad(dispatch_zeros(actual_type, r.toInt64(2), r.intlist(0)), r.toBool(3));
  }
  throw std::runtime_error("new_zeros(): invalid arguments");
}

}} // namespace torch::utils
