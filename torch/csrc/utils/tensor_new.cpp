#include "tensor_new.h"

#include <ATen/ATen.h>
#include <Python.h>

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/auto_gpu.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/python_scalars.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/tensor_numpy.h"

static const int MAX_DIMS = 128;

using namespace at;

namespace torch { namespace utils {

static Tensor new_with_sizes(const Type& type, int device, IntList sizes) {
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

static std::vector<int64_t> compute_sizes(PyObject* seq) {
  std::vector<int64_t> sizes;
  THPObjectPtr handle;
  do {
    auto length = PySequence_Length(seq);
    if (length < 0) throw python_error();
    sizes.push_back(length);
    if (sizes.size() > MAX_DIMS) {
      throw ValueError("too many dimensions '%s'", Py_TYPE(seq)->tp_name);
    }
    if (length == 0) break;
    handle = THPObjectPtr(PySequence_GetItem(seq, 0));
    seq = handle.get();
  } while (PySequence_Check(seq));

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

static Tensor new_from_sequence(ScalarType scalarType, PyObject* data) {
  if (!PySequence_Check(data)) {
    throw TypeError("new(): data must be a sequence (got %s)", Py_TYPE(data)->tp_name);
  }
  if (THPUtils_checkString(data)) {
    throw TypeError("new(): invalid data type '%s'", Py_TYPE(data)->tp_name);
  }
  if (PyArray_Check(data)) {
    return autograd::make_variable(tensor_from_numpy(data), false);
  }

  auto sizes = compute_sizes(data);
  auto tensor = autograd::make_variable(CPU(scalarType).tensor(sizes), false);
  recursive_store(
      (char*)tensor.data_ptr(), tensor.sizes(), tensor.strides(), 0,
      scalarType, tensor.type().elementSizeInBytes(), data);
  return tensor;
}

static Tensor new_from_sequence(const Type & type, int device, PyObject* data) {
  auto tensor = new_from_sequence(type.scalarType(), data);
  if (tensor.type() != type) {
    AutoNoGIL no_gil;
    AutoGPU auto_gpu(device);
    tensor = tensor.toType(type);
  }
  return tensor;
}

Tensor tensor_new(const Type& type, PyObject* args, PyObject* kwargs) {
  static PythonArgParser parser({
    "new(*, int64_t device=-1)",
    "new(IntList size, *, int64_t device=-1)",
    "new(Storage storage)",
    "new(*, int64_t cdata)|hidden",
    "new(Tensor other)",
    "new(PyObject* data, *, int64_t device=-1)",
  });

  PyObject* parsed_args[2];
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    AutoGPU auto_gpu(r.toInt64(0));
    return type.tensor();
  } else if (r.idx == 1) {
    PyObject* arg = parsed_args[0];
    if (!THPSize_Check(arg) && PyTuple_GET_SIZE(args) >= 1 && arg == PyTuple_GET_ITEM(args, 0)) {
      // new(sequence) binds to this signature but should be treated differently
      // unless the sequences is a torch.Size
      return new_from_sequence(type, r.toInt64(1), r.pyobject(0));
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
    return new_from_sequence(type, r.toInt64(1), r.pyobject(0));
  }
  throw std::runtime_error("new(): invalid arguments");
}

}} // namespace torch::utils
