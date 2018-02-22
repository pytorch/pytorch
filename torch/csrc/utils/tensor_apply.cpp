#include "tensor_apply.h"

#include <ATen/TensorUtils.h>
#include <ATen/ExpandUtils.h>

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/python_numbers.h"

using namespace at;

namespace torch { namespace utils {

static PyObject* load_scalar(void* data, ScalarType scalarType) {
  switch (scalarType) {
    case kByte: return THPUtils_packInt64(*(uint8_t*)data);
    case kChar: return THPUtils_packInt64(*(char*)data);
    case kShort: return THPUtils_packInt64(*(int16_t*)data);
    case kInt: return THPUtils_packInt64(*(int32_t*)data);
    case kLong: return THPUtils_packInt64(*(int64_t*)data);
    case kHalf: return PyFloat_FromDouble(at::convert<double, Half>(*(at::Half*)data));
    case kFloat: return PyFloat_FromDouble(*(float*)data);
    case kDouble: return PyFloat_FromDouble(*(double*)data);
    default: throw TypeError("invalid type");
  }
}

static void store_scalar(void* data, ScalarType scalarType, PyObject* obj) {
  switch (scalarType) {
    case kByte: *(uint8_t*)data = (uint8_t)THPUtils_unpackLong(obj); break;
    case kChar: *(char*)data = (char)THPUtils_unpackLong(obj); break;
    case kShort: *(int16_t*)data = (int16_t)THPUtils_unpackLong(obj); break;
    case kInt: *(int32_t*)data = (int32_t)THPUtils_unpackLong(obj); break;
    case kLong: *(int64_t*)data = THPUtils_unpackLong(obj); break;
    case kHalf: *(Half*)data = at::convert<Half, double>(THPUtils_unpackDouble(obj)); break;
    case kFloat: *(float*)data = (float)THPUtils_unpackDouble(obj); break;
    case kDouble: *(double*)data = THPUtils_unpackDouble(obj); break;
    default: throw TypeError("invalid type");
  }
}

struct StridedData {
  StridedData(const Tensor & tensor)
    : data(tensor.data_ptr())
    , strides(tensor.strides())
    , elementSize(tensor.type().elementSizeInBytes()) {}

  void* data;
  IntList strides;
  int64_t elementSize;

  void step(int dim) {
    data = (char*)data + (strides[dim] * elementSize);
  }
};

template<size_t N>
static void recursive_apply(IntList sizes, ScalarType scalarType, int64_t dim,
                            PyObject* fn, std::array<StridedData, N> strided_data) {
  int64_t ndim = sizes.size();
  if (dim == ndim) {
    auto args = THPObjectPtr(PyTuple_New(N));
    if (!args) throw python_error();
    for (size_t i = 0; i < N; i++) {
      PyObject* arg = load_scalar(strided_data[i].data, scalarType);
      if (!arg) throw python_error();
      PyTuple_SET_ITEM(args.get(), i, arg);
    }
    auto ret = THPObjectPtr(PyObject_CallObject(fn, args.get()));
    if (!ret) throw python_error();
    store_scalar(strided_data[0].data, scalarType, ret.get());
    return;
  }

  auto n = sizes[dim];
  for (int64_t i = 0; i < n; i++) {
    recursive_apply(sizes, scalarType, dim + 1, fn, strided_data);
    for (auto& td : strided_data) {
      td.step(dim);
    }
  }
}

Tensor & apply_(Tensor & self, PyObject* fn) {
  if (self.type().backend() != kCPU) {
    throw TypeError("apply_ is only implemented on CPU tensors");
  }
  auto scalarType = self.type().scalarType();
  recursive_apply<1>(self.sizes(), scalarType, 0, fn, {{ self }});
  return self;
}

Tensor & map_(Tensor & self, const Tensor & other_, PyObject* fn) {
  if (self.type().backend() != kCPU) {
    throw TypeError("map_ is only implemented on CPU tensors");
  }
  if (other_.type() != self.type()) {
    throw TypeError("map_: expected %s for 'other' (got %s)",
        self.type().toString(), other_.type().toString());
  }
  Tensor other;
  std::tie(other) = expand_inplace(self, other_, "map_");
  auto scalarType = self.type().scalarType();
  recursive_apply<2>(self.sizes(), scalarType, 0, fn, {{ self, other }});
  return self;
}

Tensor & map2_(Tensor & self, const Tensor & x_, const Tensor & y_, PyObject* fn) {
  if (self.type().backend() != kCPU || x_.type().backend() != kCPU || y_.type().backend() != kCPU) {
    throw TypeError("map2_ is only implemented on CPU tensors");
  }
  if (x_.type() != self.type()) {
    throw TypeError("map2_: expected %s for argument 'x' (got %s)",
        self.type().toString(), x_.type().toString());
  }
  if (y_.type() != self.type()) {
    throw TypeError("map2_: expected %s for argument 'y' (got %s)",
        self.type().toString(), y_.type().toString());
  }
  Tensor other1, other2;
  std::tie(other1, other2) = expand_inplace(self, x_, y_, "map2_");
  auto scalarType = self.type().scalarType();
  recursive_apply<3>(self.sizes(), scalarType, 0, fn, {{ self, other1, other2 }});
  return self;
}

}} // namespace torch::utils
