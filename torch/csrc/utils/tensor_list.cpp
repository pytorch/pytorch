#include <torch/csrc/utils/tensor_list.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/python_scalars.h>

using namespace at;

namespace torch { namespace utils {

static PyObject* recursive_to_list(
    char* data, IntArrayRef sizes, IntArrayRef strides, int64_t dim,
    ScalarType scalarType, int64_t elementSize)
{
  int64_t ndim = sizes.size();
  if (dim == ndim) {
    return torch::utils::load_scalar(data, scalarType);
  }
  auto n = sizes[dim];
  auto list = THPObjectPtr(PyList_New(n));
  if (!list) throw python_error();
  for (int64_t i = 0; i < n; i++) {
    PyObject* obj = recursive_to_list(data, sizes, strides, dim + 1, scalarType, elementSize);
    if (!obj) throw python_error();
    PyList_SET_ITEM(list.get(), i, obj);
    data += strides[dim] * elementSize;
  }
  return list.release();
}

PyObject* tensor_to_list(const Tensor& tensor) {
  Tensor data = tensor;
  if (data.type().backend() != Backend::CPU) {
    with_no_gil([&]() {
      data = data.toBackend(Backend::CPU);
    });
  }
  auto& type = data.type();
  return recursive_to_list(
      (char*)data.data_ptr(), data.sizes(), data.strides(), 0,
      type.scalarType(), type.elementSizeInBytes());
}

}}  // namespace torch::utils
