#include "tensor_list.h"

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_numbers.h"

using namespace at;

namespace torch {

static PyObject* toList(char* data, IntList sizes, IntList strides, int64_t dim,
                        ScalarType scalarType, int64_t elementSize)
{
  int64_t ndim = sizes.size();
  if (dim == ndim) {
    switch (scalarType) {
      case kByte: return THPUtils_packInt64(*(uint8_t*)data);
      case kChar: return THPUtils_packInt64(*(char*)data);
      case kShort: return THPUtils_packInt64(*(int16_t*)data);
      case kInt: return THPUtils_packInt64(*(int32_t*)data);
      case kLong: return THPUtils_packInt64(*(int64_t*)data);
      case kHalf: return PyFloat_FromDouble(at::convert<double, Half>(*(at::Half*)data));
      case kFloat: return PyFloat_FromDouble(*(float*)data);
      case kDouble: return PyFloat_FromDouble(*(double*)data);
      default: throw std::runtime_error("invalid type");
    }
  }
  auto n = sizes[dim];
  auto list = THPObjectPtr(PyList_New(n));
  if (!list) throw python_error();
  for (int64_t i = 0; i < n; i++) {
    PyObject* obj = toList(data, sizes, strides, dim + 1, scalarType, elementSize);
    if (!obj) throw python_error();
    PyList_SET_ITEM(list.get(), i, obj);
    data += strides[dim] * elementSize;
  }
  return list.release();
}

PyObject* THPUtils_tensorToList(const Tensor& tensor) {
  Tensor data = tensor;
  if (data.type().backend() != kCPU) {
    with_no_gil([&]() {
      data = data.toBackend(kCPU);
    });
  }
  auto& type = data.type();
  return toList(
      (char*)data.data_ptr(), data.sizes(), data.strides(), 0,
      type.scalarType(), type.elementSizeInBytes());
}

}
