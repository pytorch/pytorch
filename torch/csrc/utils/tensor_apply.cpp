#include <torch/csrc/utils/tensor_apply.h>

#include <ATen/ExpandUtils.h>
#include <ATen/TensorUtils.h>
#include <c10/util/irange.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_scalars.h>

using namespace at;

namespace torch::utils {

struct StridedData {
  StridedData(const Tensor& tensor)
      : data(tensor.data_ptr()),
        strides(tensor.strides()),
        elementSize(tensor.element_size()) {}

  void* data;
  IntArrayRef strides;
  int64_t elementSize;

  void step(int dim) {
    data = (char*)data + (strides[dim] * elementSize);
  }
};

template <size_t N>
static void recursive_apply(
    IntArrayRef sizes,
    ScalarType scalarType,
    int64_t dim,
    PyObject* fn,
    std::array<StridedData, N> strided_data) {
  int64_t ndim = static_cast<int64_t>(sizes.size());
  if (dim == ndim) {
    auto args = THPObjectPtr(PyTuple_New(N));
    if (!args)
      throw python_error();
    for (const auto i : c10::irange(N)) {
      PyObject* arg = load_scalar(strided_data[i].data, scalarType);
      if (!arg)
        throw python_error();
      PyTuple_SET_ITEM(args.get(), i, arg);
    }
    auto ret = THPObjectPtr(PyObject_CallObject(fn, args.get()));
    if (!ret)
      throw python_error();
    store_scalar(strided_data[0].data, scalarType, ret.get());
    return;
  }

  auto n = sizes[dim];
  for ([[maybe_unused]] const auto i : c10::irange(n)) {
    recursive_apply(sizes, scalarType, dim + 1, fn, strided_data);
    for (auto& td : strided_data) {
      td.step(dim);
    }
  }
}

const Tensor& apply_(const Tensor& self, PyObject* fn) {
  if (self.is_meta()) {
    return self; // Just skip
  }
  TORCH_CHECK_TYPE(
      self.device().is_cpu(), "apply_ is only implemented on CPU tensors");
  auto scalarType = self.scalar_type();
  recursive_apply<1>(self.sizes(), scalarType, 0, fn, {{self}});
  return self;
}

const Tensor& map_(const Tensor& self, const Tensor& other_, PyObject* fn) {
  TORCH_CHECK_TYPE(
      other_.options().type_equal(self.options()),
      "map_: expected ",
      self.toString(),
      " for 'other' (got ",
      other_.toString(),
      ")");
  if (self.is_meta()) {
    return self; // Just skip
  }
  TORCH_CHECK_TYPE(
      self.device().is_cpu(), "map_ is only implemented on CPU tensors");
  c10::MaybeOwned<Tensor> other = expand_inplace(self, other_, "map_");
  auto scalarType = self.scalar_type();
  recursive_apply<2>(self.sizes(), scalarType, 0, fn, {{self, *other}});
  return self;
}

const Tensor& map2_(
    const Tensor& self,
    const Tensor& x_,
    const Tensor& y_,
    PyObject* fn) {
  TORCH_CHECK_TYPE(
      x_.options().type_equal(self.options()),
      "map2_: expected ",
      self.toString(),
      " for argument 'x' (got ",
      x_.toString(),
      ")");
  TORCH_CHECK_TYPE(
      y_.options().type_equal(self.options()),
      "map2_: expected ",
      self.toString(),
      " for argument 'y' (got ",
      y_.toString(),
      ")");
  if (self.is_meta()) {
    return self; // Just skip
  }
  TORCH_CHECK_TYPE(
      (self.device().is_cpu() && x_.device().is_cpu() && y_.device().is_cpu()),
      "map2_ is only implemented on CPU tensors");
  auto others = expand_inplace(self, x_, y_, "map2_");
  auto scalarType = self.scalar_type();
  recursive_apply<3>(
      self.sizes(),
      scalarType,
      0,
      fn,
      {{self, *std::get<0>(others), *std::get<1>(others)}});
  return self;
}

} // namespace torch::utils
