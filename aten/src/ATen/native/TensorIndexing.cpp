#include <ATen/native/TensorIndexing.h>

#include <c10/util/Exception.h>

namespace at {
namespace indexing {

const EllipsisIndexType Ellipsis = EllipsisIndexType();

std::ostream& operator<<(std::ostream& stream, const Slice& slice) {
  stream << slice.start() << ":" << slice.stop() << ":" << slice.step();
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const TensorIndex& tensor_index) {
  if (tensor_index.is_none()) {
    stream << "None";
  } else if (tensor_index.is_ellipsis()) {
    stream << "...";
  } else if (tensor_index.is_integer()) {
    stream << tensor_index.integer();
  } else if (tensor_index.is_boolean()) {
    stream << std::boolalpha << tensor_index.boolean();
  } else if (tensor_index.is_slice()) {
    stream << tensor_index.slice();
  } else if (tensor_index.is_tensor()) {
    stream << tensor_index.tensor();
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const std::vector<TensorIndex>& tensor_indices) {
  stream << "(";
  for (size_t i = 0; i < tensor_indices.size(); i++) {
    stream << tensor_indices[i];
    if (i < tensor_indices.size() - 1) stream << ", ";
  }
  stream << ")";
  return stream;
}

// This mirrors `THPVariable_setitem` in torch/csrc/autograd/python_variable_indexing.cpp
// for "the assigned value is a Tensor" case
static inline void set_item(Tensor& self, ArrayRef<TensorIndex> indices, const Tensor& value, bool is_tracing) {
  OptionalDeviceGuard device_guard(device_of(self));
  at::Device self_device = self.device();
  IntArrayRef self_sizes = self.sizes();

  // handle simple types: integers, slices, ellipsis, bool
  if (indices.size() == 1) {
    const TensorIndex& index = indices[0];
    if (index.is_boolean() && !index.boolean()) {
      // do nothing for false (technically we should check the size, but we don't have
      // real 0-sized shapes.
      return;
    } else if (index.is_ellipsis()) {
      copy_to(self, value);
      return;
    } else if (index.is_none() || (index.is_boolean() && index.boolean())) {
      copy_to(self.unsqueeze(0), value);
      return;
    } else if (index.is_integer()) {
      copy_to(applySelect(self, 0, index.integer(), 0, self_device, self_sizes), value);
      return;
    } else if (index.is_slice()) {
      copy_to(applySlice(
        self,
        0,
        index.slice().start(),
        index.slice().stop(),
        index.slice().step(),
        /*ensure_view=*/false,
        /*is_tracing=*/is_tracing,
        self_device,
        self_sizes), value);
      return;
    }
  }

  std::vector<Tensor> tensorIndices;
  Tensor sliced = applySlicing(self, indices, tensorIndices, is_tracing, self_device, self_sizes);
  if (tensorIndices.empty()) {
    copy_to(sliced, value);
    return;
  }

  IntArrayRef valueSizes = value.sizes();
  IntArrayRef slicedValueSizes = slicePrefix1sSize(valueSizes);
  Tensor valuesSliced;
  if (!valueSizes.equals(slicedValueSizes)) {
    valuesSliced = value.view(slicedValueSizes);
  } else {
    valuesSliced = value;
  }
  dispatch_index_put_(sliced, std::move(tensorIndices), valuesSliced);
  return;
}

// This mirrors `THPVariable_setitem` in torch/csrc/autograd/python_variable_indexing.cpp
// for "the assigned value is a Scalar" case
static inline void set_item(Tensor& self, ArrayRef<TensorIndex> indices, Scalar v, bool is_tracing) {
  OptionalDeviceGuard device_guard(device_of(self));
  Tensor value;

  {
    at::AutoNonVariableTypeMode guard;
    // TODO: This qint special case looks very suspicious...
    if (isQIntType(self.scalar_type())) {
      value = at::indexing::scalarToTensor(v, device(kCPU).dtype(kFloat), at::Device(kCPU));
    } else {
      value = at::indexing::scalarToTensor(v, self.options(), self.device());
    }
  }

  return set_item(self, indices, value, is_tracing);
}

} // namespace indexing

Tensor Tensor::index(ArrayRef<at::indexing::TensorIndex> indices) const {
  return at::indexing::get_item(*this, indices, /*is_tracing=*/false);
}
Tensor Tensor::index(std::initializer_list<at::indexing::TensorIndex> indices) const {
  return index(ArrayRef<at::indexing::TensorIndex>(indices));
}

Tensor & Tensor::index_put_(ArrayRef<at::indexing::TensorIndex> indices, Tensor const & rhs) {
  at::indexing::set_item(*this, indices, rhs, /*is_tracing=*/false);
  return *this;
}
Tensor & Tensor::index_put_(ArrayRef<at::indexing::TensorIndex> indices, Scalar v) {
  at::indexing::set_item(*this, indices, v, /*is_tracing=*/false);
  return *this;
}
Tensor & Tensor::index_put_(std::initializer_list<at::indexing::TensorIndex> indices, Tensor const & rhs) {
  return index_put_(ArrayRef<at::indexing::TensorIndex>(indices), rhs);
}
Tensor & Tensor::index_put_(std::initializer_list<at::indexing::TensorIndex> indices, Scalar v) {
  return index_put_(ArrayRef<at::indexing::TensorIndex>(indices), v);
}

} // namespace at
