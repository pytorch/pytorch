#include <torch/csrc/utils/tensor_apply.h>

#include <ATen/TensorUtils.h>
#include <ATen/ExpandUtils.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_scalars.h>

using namespace at;

namespace torch { namespace utils {

struct StridedData {
  StridedData(const Tensor & tensor)
    : data(tensor.data_ptr())
    , strides(tensor.strides())
    , elementSize(tensor.element_size()) {}

  void* data;
  IntArrayRef strides;
  int64_t elementSize;

  void step(int dim) {
    data = (char*)data + (strides[dim] * elementSize);
  }
};

template<size_t N>
static void recursive_apply(IntArrayRef sizes, ScalarType scalarType, int64_t dim,
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
  if (!self.device().is_cpu()) {
    throw TypeError("apply_ is only implemented on CPU tensors");
  }
  auto scalarType = self.scalar_type();
  recursive_apply<1>(self.sizes(), scalarType, 0, fn, {{ self }});
  return self;
}

Tensor & map_(Tensor & self, const Tensor & other_, PyObject* fn) {
  if (!self.device().is_cpu()) {
    throw TypeError("map_ is only implemented on CPU tensors");
  }
  if (!other_.options().type_equal(self.options())) {
    throw TypeError("map_: expected %s for 'other' (got %s)",
        self.toString().c_str(), other_.toString().c_str());
  }
  Tensor other;
  std::tie(other) = expand_inplace(self, other_, "map_");
  auto scalarType = self.scalar_type();
  recursive_apply<2>(self.sizes(), scalarType, 0, fn, {{ self, other }});
  return self;
}

Tensor & map2_(Tensor & self, const Tensor & x_, const Tensor & y_, PyObject* fn) {
  if (!self.device().is_cpu() || !x_.device().is_cpu() || !y_.device().is_cpu()) {
    throw TypeError("map2_ is only implemented on CPU tensors");
  }
  if (!x_.options().type_equal(self.options())) {
    throw TypeError("map2_: expected %s for argument 'x' (got %s)",
        self.toString().c_str(), x_.toString().c_str());
  }
  if (!y_.options().type_equal(self.options())) {
    throw TypeError("map2_: expected %s for argument 'y' (got %s)",
        self.toString().c_str(), y_.toString().c_str());
  }
  Tensor other1, other2;
  std::tie(other1, other2) = expand_inplace(self, x_, y_, "map2_");
  auto scalarType = self.scalar_type();
  recursive_apply<3>(self.sizes(), scalarType, 0, fn, {{ self, other1, other2 }});
  return self;
}

}} // namespace torch::utils
