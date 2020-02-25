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
    c10::optional<int64_t> start_index = at::indexing::None,
    c10::optional<int64_t> stop_index = at::indexing::None,
    c10::optional<int64_t> step_index = at::indexing::None) {
  int64_t start, stop, step;
  if (!step_index.has_value()) {
    step = 1;
  } else {
    step = step_index.value();
    if (step == 0) {
      TORCH_CHECK(false, "slice step cannot be zero");
    }
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
  TORCH_CHECK(
    strcmp(str, "...") == 0,
    "Expected \"...\" to represent an ellipsis index, but got \"", str, "\"");
}
TensorIndex::TensorIndex(int64_t integer) : integer_(integer), type_(TensorIndexType::Integer) {}
TensorIndex::TensorIndex(int integer) : TensorIndex((int64_t)integer) {}
TensorIndex::TensorIndex(std::initializer_list<c10::optional<int64_t>> init_list)
    : type_(TensorIndexType::Slice) {
  if (init_list.size() == 0) {
    slice_ = unpackSlice();
  } else if (init_list.size() == 2) {
    slice_ = unpackSlice(*init_list.begin(), *(init_list.begin() + 1));
  } else if (init_list.size() == 3) {
    slice_ = unpackSlice(*init_list.begin(), *(init_list.begin() + 1), *(init_list.begin() + 2));
  } else {
    TORCH_CHECK(
      false,
      "Expected 0 / 2 / 3 elements in the braced-init-list to represent a slice index, but got ",
      init_list.size(),
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

} // namespace indexing
} // namespace at
