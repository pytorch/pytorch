#include <ATen/native/TensorIndexing.h>

#include <c10/util/Exception.h>

namespace at {
namespace indexing {

const EllipsisIndexType Ellipsis = EllipsisIndexType();

Slice::Slice() {}
Slice::Slice(int64_t start, int64_t stop, int64_t step) : start_(start), stop_(stop), step_(step) {}

int64_t Slice::start() const {
  return start_;
}

int64_t Slice::stop() const {
  return stop_;
}

int64_t Slice::step() const {
  return step_;
}

std::ostream& operator<<(std::ostream& stream, const Slice& slice) {
  stream << slice.start() << ":" << slice.stop() << ":" << slice.step();
  return stream;
}

// This mirrors `__PySlice_Unpack` in torch/csrc/utils/python_compat.h
Slice unpackSlice(
    c10::optional<int64_t> start_index,
    c10::optional<int64_t> stop_index,
    c10::optional<int64_t> step_index) {
  int64_t start, stop, step;
  if (!step_index.has_value()) {
    step = 1;
  } else {
    step = step_index.value();
    TORCH_CHECK_VALUE(step != 0, "slice step cannot be zero");

    // Here step might be -INDEX_MAX-1; in this case we replace it
    // with -INDEX_MAX.  This doesn't affect the semantics, and it
    // guards against later undefined behaviour resulting from code that
    // does "step = -step" as part of a slice reversal.
    if (step < -INDEX_MAX)
      step = -INDEX_MAX;
  }
  if (!start_index.has_value()) {
    start = step < 0 ? INDEX_MAX : 0;
  } else {
    start = start_index.value();
  }
  if (!stop_index.has_value()) {
    stop = step < 0 ? INDEX_MIN : INDEX_MAX;
  } else {
    stop = stop_index.value();
  }
  return Slice(start, stop, step);
}

TensorIndex::TensorIndex(c10::nullopt_t) : type_(TensorIndexType::None) {}
TensorIndex::TensorIndex(at::indexing::EllipsisIndexType) : type_(TensorIndexType::Ellipsis) {}
TensorIndex::TensorIndex(const char *str) : TensorIndex(at::indexing::Ellipsis) {
  TORCH_CHECK_VALUE(
    strcmp(str, "...") == 0,
    "Expected \"...\" to represent an ellipsis index, but got \"", str, "\"");
}
TensorIndex::TensorIndex(int64_t integer) : integer_(integer), type_(TensorIndexType::Integer) {}
TensorIndex::TensorIndex(int integer) : TensorIndex((int64_t)integer) {}
TensorIndex::TensorIndex(std::initializer_list<c10::optional<int64_t>> slice) : type_(TensorIndexType::Slice) {
  if (slice.size() == 0) {
    slice_ = unpackSlice(c10::nullopt, c10::nullopt, c10::nullopt);
  } else if (slice.size() == 2) {
    slice_ = unpackSlice(*slice.begin(), *(slice.begin() + 1), c10::nullopt);
  } else if (slice.size() == 3) {
    slice_ = unpackSlice(*slice.begin(), *(slice.begin() + 1), *(slice.begin() + 2));
  } else {
    TORCH_CHECK_VALUE(
      false,
      "Expected 0 / 2 / 3 elements in the braced-init-list to represent a slice index, but got ",
      slice.size(),
      " element(s)");
  }
}
TensorIndex::TensorIndex(Tensor tensor) : tensor_(tensor), type_(TensorIndexType::Tensor) {}

bool TensorIndex::is_none() const {
  return type_ == TensorIndexType::None;
}

bool TensorIndex::is_ellipsis() const {
  return type_ == TensorIndexType::Ellipsis;
}

bool TensorIndex::is_integer() const {
  return type_ == TensorIndexType::Integer;
}

int64_t TensorIndex::integer() const {
  return integer_;
}

bool TensorIndex::is_boolean() const {
  return type_ == TensorIndexType::Boolean;
}

bool TensorIndex::boolean() const {
  return boolean_;
}

bool TensorIndex::is_slice() const {
  return type_ == TensorIndexType::Slice;
}

const Slice& TensorIndex::slice() const {
  return slice_;
}

bool TensorIndex::is_tensor() const {
  return type_ == TensorIndexType::Tensor;
}

const Tensor& TensorIndex::tensor() const {
  return tensor_;
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

// This mirrors `count_specified_dimensions` in torch/csrc/autograd/python_variable_indexing.cpp
int64_t count_specified_dimensions(ArrayRef<TensorIndex> indices) {
  // Count the number of indexed dimensions (everything but ellipsis and None)
  int64_t count = 0;
  size_t size = indices.size();
  for (size_t i = 0; i < size; i++) {
    auto& obj = indices[i];
    if (obj.is_tensor()) {
      auto& tensor = obj.tensor();
      if (tensor.scalar_type() == kByte || tensor.scalar_type() == kBool) {
        count += tensor.dim();
      } else {
        count++;
      }
    } else if (!obj.is_none() && !obj.is_ellipsis() && !obj.is_boolean()) {
      count++;
    }
  }
  return count;
}

// This mirrors `applySlicing` in torch/csrc/autograd/python_variable_indexing.cpp
Tensor applySlicing(const Tensor& self, ArrayRef<TensorIndex> indices, std::vector<Tensor>& outIndices) {
  int64_t size = indices.size();
  int64_t dim = 0;
  int64_t specified_dims = count_specified_dimensions(indices);

  TORCH_CHECK_INDEX(specified_dims <= self.dim(), "too many indices for tensor of dimension ", (int)self.dim());

  Tensor result = self;
  for (int64_t i = 0; i < size; i++) {
    auto& obj = indices[i];
    if (obj.is_integer()) {
      result = handleInteger(result, dim, obj.integer(), i);
    } else if (obj.is_slice()) {
      result = handleSlice(result, dim, obj.slice().start(), obj.slice().stop(), obj.slice().step());
    } else if (obj.is_ellipsis()) {
      handleEllipsis(self, dim, specified_dims);
    } else if (obj.is_none()) {
      result = handleNone(result, dim);
    } else if (obj.is_boolean()) {
      result = handleBoolean(result, obj.boolean(), outIndices, dim);
    } else if (obj.is_tensor()) {
      result = handleTensor(result, obj.tensor(), outIndices, dim, i);
    } else {
      TORCH_INTERNAL_ASSERT(false, "Invalid TensorIndex type");
    }
  }
  return result;
}

// This mirrors `THPVariable_getitem` in torch/csrc/autograd/python_variable_indexing.cpp
Tensor get_item(const Tensor& self, ArrayRef<TensorIndex> indices) {
  OptionalDeviceGuard device_guard(device_of(self));

  // handle simple types: integers, slices, ellipsis
  if (indices.size() == 1) {
    const TensorIndex& index = indices[0];
    if (index.is_none()) {
      return handleNoneSingleDim(self);
    } else if (index.is_ellipsis()) {
      return handleEllipsisSingleDim(self);
    } else if (index.is_integer()) {
      return handleIntegerSingleDim(self, index.integer());
    } else if (index.is_slice()) {
      return handleSliceSingleDim(
        self,
        index.slice().start(),
        index.slice().stop(),
        index.slice().step(),
        true);
    }
  }

  std::vector<Tensor> tensorIndices;
  Tensor sliced = applySlicing(self, indices, tensorIndices);
  if (tensorIndices.empty()) {
    if (sliced.is_same(self)) {
      // ensure we return a shallow copy for things like x[...]
      sliced = sliced.alias();
    }
    return sliced;
  }

  // indexing by tensors ("advanced" indexing)
  return dispatch_index(sliced, tensorIndices);
}

// This mirrors `THPVariable_setitem` in torch/csrc/autograd/python_variable_indexing.cpp
// for "the assigned value is a Tensor" case
void set_item(Tensor& self, ArrayRef<TensorIndex> indices, const Tensor& value) {
  OptionalDeviceGuard device_guard(device_of(self));

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
      copy_to(handleNoneSingleDim(self), value);
      return;
    } else if (index.is_integer()) {
      copy_to(handleIntegerSingleDim(self, index.integer()), value);
      return;
    } else if (index.is_slice()) {
      copy_to(handleSliceSingleDim(
        self,
        index.slice().start(),
        index.slice().stop(),
        index.slice().step(),
        false), value);
      return;
    }
  }

  std::vector<Tensor> tensorIndices;
  Tensor sliced = applySlicing(self, indices, tensorIndices);
  if (tensorIndices.empty()) {
    copy_to(sliced, value);
    return;
  }

  IntArrayRef slicedValueSizes = slicePrefix1sSize(value.sizes());
  Tensor valuesSliced;
  if (!value.sizes().equals(slicedValueSizes)) {
    valuesSliced = value.view(slicedValueSizes);
  } else {
    valuesSliced = value;
  }
  dispatch_index_put_(sliced, tensorIndices, valuesSliced);
  return;
}

// This mirrors `THPVariable_setitem` in torch/csrc/autograd/python_variable_indexing.cpp
// for "the assigned value is a Scalar" case
void set_item(Tensor& self, ArrayRef<TensorIndex> indices, Scalar v) {
  OptionalDeviceGuard device_guard(device_of(self));
  Tensor value;

  // TODO: This qint special case looks very suspicious...
  if (isQIntType(self.scalar_type())) {
    value = at::native::scalar_tensor(v, device(kCPU).dtype(kFloat));
  } else {
    value = at::native::scalar_tensor(v, self.options());
  }

  return set_item(self, indices, value);
}

} // namespace indexing

Tensor Tensor::index(ArrayRef<at::indexing::TensorIndex> indices) const {
  return at::indexing::get_item(*this, indices);
}
Tensor Tensor::index(std::initializer_list<at::indexing::TensorIndex> indices) const {
  return index(ArrayRef<at::indexing::TensorIndex>(indices));
}

Tensor & Tensor::index_put_(ArrayRef<at::indexing::TensorIndex> indices, Tensor const & rhs) {
  at::indexing::set_item(*this, indices, rhs);
  return *this;
}
Tensor & Tensor::index_put_(ArrayRef<at::indexing::TensorIndex> indices, Tensor && rhs) {
  at::indexing::set_item(*this, indices, rhs);
  return *this;
}
Tensor & Tensor::index_put_(ArrayRef<at::indexing::TensorIndex> indices, Scalar v) {
  at::indexing::set_item(*this, indices, v);
  return *this;
}
Tensor & Tensor::index_put_(std::initializer_list<at::indexing::TensorIndex> indices, Tensor const & rhs) {
  return index_put_(ArrayRef<at::indexing::TensorIndex>(indices), rhs);
}
Tensor & Tensor::index_put_(std::initializer_list<at::indexing::TensorIndex> indices, Tensor && rhs) {
  return index_put_(ArrayRef<at::indexing::TensorIndex>(indices), rhs);
}
Tensor & Tensor::index_put_(std::initializer_list<at::indexing::TensorIndex> indices, Scalar v) {
  return index_put_(ArrayRef<at::indexing::TensorIndex>(indices), v);
}

} // namespace at
