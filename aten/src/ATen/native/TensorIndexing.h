#pragma once

#include <c10/util/Optional.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>

namespace at {
namespace indexing {

const int64_t INDEX_MAX = std::numeric_limits<int64_t>::max();
const int64_t INDEX_MIN = std::numeric_limits<int64_t>::min();

enum class TensorIndexType { None, Ellipsis, Integer, Boolean, Slice, Tensor };

constexpr c10::nullopt_t None{c10::nullopt_t::init()};

struct CAFFE2_API EllipsisIndexType final { EllipsisIndexType() {} };
CAFFE2_API extern const EllipsisIndexType Ellipsis;

struct CAFFE2_API Slice final {
 public:
  Slice() {}
  Slice(int64_t start, int64_t stop, int64_t step) : start_(start), stop_(stop), step_(step) {}

  inline int64_t start() const {
    return start_;
  }

  inline int64_t stop() const {
    return stop_;
  }

  inline int64_t step() const {
    return step_;
  }

 private:
  int64_t start_;
  int64_t stop_;
  int64_t step_;
};

CAFFE2_API std::ostream& operator<<(std::ostream& stream, const Slice& slice);

// This mirrors `__PySlice_Unpack` in torch/csrc/utils/python_compat.h
inline Slice unpackSlice(
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

// `at::indexing::TensorIndex` is used for converting C++ tensor indices such as
// `{None, "...", Ellipsis, 0, true, {1, None, 2}, torch::tensor({1, 2})}`
// into its equivalent `std::vector<TensorIndex>`, so that further tensor indexing
// operations can be performed using the supplied indices.
//
// There is one-to-one correspondence between Python and C++ tensor index types:
// Python                  | C++
// -----------------------------------------------------
// `None`                  | `at::indexing::None`
// `Ellipsis`              | `at::indexing::Ellipsis`
// `...`                   | `"..."`
// `123`                   | `123`
// `True` / `False`        | `true` / `false`
// `:`                     | `{}` / `{None, None}`
// `::`                    | `{}` / `{None, None, None}`
// `1:`                    | `{1, None}`
// `1::`                   | `{1, None, None}`
// `:3`                    | `{None, 3}`
// `:3:`                   | `{None, 3, None}`
// `::2`                   | `{None, None, 2}`
// `1:3`                   | `{1, 3}`
// `1::2`                  | `{1, None, 2}`
// `:3:2`                  | `{None, 3, 2}`
// `1:3:2`                 | `{1, 3, 2}`
// `torch.tensor([1, 2])`) | `torch::tensor({1, 2})`
struct CAFFE2_API TensorIndex final {
  // Case 1: `at::indexing::None`
  TensorIndex(c10::nullopt_t) : type_(TensorIndexType::None) {}

  // Case 2: "..." / `at::indexing::Ellipsis`
  TensorIndex(at::indexing::EllipsisIndexType) : type_(TensorIndexType::Ellipsis) {}
  TensorIndex(const char *str) : TensorIndex(at::indexing::Ellipsis) {
    TORCH_CHECK_VALUE(
      strcmp(str, "...") == 0,
      "Expected \"...\" to represent an ellipsis index, but got \"", str, "\"");
  }

  // Case 3: Integer value
  TensorIndex(int64_t integer) : integer_(integer), type_(TensorIndexType::Integer) {}
  TensorIndex(int integer) : TensorIndex((int64_t)integer) {}

  // Case 4: Boolean value
  template <class T,
            class = typename std::enable_if<std::is_same<bool, T>::value>::type >
  TensorIndex(T boolean) : boolean_(boolean), type_(TensorIndexType::Boolean) {}

  // Case 5: Slice represented in `{start, stop, step}` form,
  // where `start` / `stop` / `step` can be integer or `at::indexing::None`
  TensorIndex(std::initializer_list<c10::optional<int64_t>> slice) : type_(TensorIndexType::Slice) {
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

  // Case 5: Tensor value
  TensorIndex(Tensor tensor) : tensor_(tensor), type_(TensorIndexType::Tensor) {}

  inline bool is_none() const {
    return type_ == TensorIndexType::None;
  }

  inline bool is_ellipsis() const {
    return type_ == TensorIndexType::Ellipsis;
  }

  inline bool is_integer() const {
    return type_ == TensorIndexType::Integer;
  }

  inline int64_t integer() const {
    return integer_;
  }

  inline bool is_boolean() const {
    return type_ == TensorIndexType::Boolean;
  }

  inline bool boolean() const {
    return boolean_;
  }

  inline bool is_slice() const {
    return type_ == TensorIndexType::Slice;
  }

  inline const Slice& slice() const {
    return slice_;
  }

  inline bool is_tensor() const {
    return type_ == TensorIndexType::Tensor;
  }

  inline const Tensor& tensor() const {
    return tensor_;
  }

 private:
  int64_t integer_;
  bool boolean_;
  Slice slice_;
  Tensor tensor_;
  TensorIndexType type_;
};

CAFFE2_API std::ostream& operator<<(std::ostream& stream, const TensorIndex& tensor_index);
CAFFE2_API std::ostream& operator<<(std::ostream& stream, const std::vector<TensorIndex>& tensor_indices);

static inline Tensor applySlice(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t stop,
    int64_t step,
    bool ensure_view,
    bool is_tracing,
    const at::Device& self_device,
    const IntArrayRef& self_sizes) {
  // TODO: implement negative step
  TORCH_CHECK_VALUE(step > 0, "step must be greater than zero");

  // Skip this optimization if we are tracing, as the trace may be polymorphic
  // over the shape of the `self` tensor, and we still want to record
  // the slice.
  int64_t length = (self_device == at::kCPU || self_device == at::kCUDA) ? self_sizes[dim] : self.size(dim);
  if (!ensure_view && start == 0 && stop == length && step == 1 && !is_tracing) {
    return self;
  }
  return self.slice(dim, start, stop, step);
}

static inline Tensor applySelect(
    const Tensor& self,
    int64_t dim,
    int64_t index,
    int64_t real_dim,
    const at::Device& self_device,
    const IntArrayRef& self_sizes) {
  TORCH_CHECK_INDEX(
    !(index == 0 && dim == 0 && self_sizes.size() == 0),
    "invalid index of a 0-dim tensor. ",
    "Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number");

  int64_t size = self_sizes[dim];
  TORCH_CHECK_INDEX(
    index >= -size && index < size,
    "index ", index, " is out of bounds for dimension ", real_dim, " with size ", size);

  // if the index is negative, do not normalize it because that would fix the index
  // on the current tensor size in the tracer.
  // aten::select also works on negative indices
  return self.select(dim, index);
}

static inline Tensor boolToIndexingTensorCPUOrCUDA(const Tensor& self, bool value) {
  // booleans add a dimension of size 1. true indexes this dimension as if 0:, false as empty.
  if (value) {
    return at::native::zeros({1}, {}, self.options().dtype(kLong));
  } else {
    return at::native::empty({0}, {}, self.options().dtype(kLong));
  }
}

static inline Tensor boolToIndexingTensorNonNativeDeviceType(const Tensor& self, bool value) {
  // booleans add a dimension of size 1. true indexes this dimension as if 0:, false as empty.
  if (value) {
    return at::zeros({1}, {}, self.options().dtype(kLong));
  } else {
    return at::empty({0}, {}, self.options().dtype(kLong));
  }
}

static inline Tensor boolToIndexingTensor(const Tensor& self, bool value, const at::Device& self_device) {
  if (self_device == at::kCPU || self_device == at::kCUDA) {
    return boolToIndexingTensorCPUOrCUDA(self, value);
  } else {
    return boolToIndexingTensorNonNativeDeviceType(self, value);
  }
}

static inline Tensor scalarToTensorCPUOrCUDA(Scalar v, const TensorOptions& options) {
  return at::native::scalar_tensor(v, options);
}

static inline Tensor scalarToTensorNonNativeDeviceType(Scalar v, const TensorOptions& options) {
  return at::scalar_tensor(v, options);
}

static inline Tensor scalarToTensor(Scalar v, const TensorOptions& options, const at::Device& self_device) {
  if (self_device == at::kCPU || self_device == at::kCUDA) {
    return scalarToTensorCPUOrCUDA(v, options);
  } else {
    return scalarToTensorNonNativeDeviceType(v, options);
  }
}

// To match numpy semantics:
// As a special case for backwards compatibility,
// strip away unit dimensions from the left of 'src'
static inline IntArrayRef slicePrefix1sSize(const IntArrayRef& sizes) {
  size_t first_non1_src = sizes.size();
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] != 1) {
      first_non1_src = i;
      break;
    }
  }

  return sizes.slice(first_non1_src);
}

static inline void copy_to(Tensor dst, const Tensor& src) {
  Tensor b_src;
  IntArrayRef sliced_src_sizes = slicePrefix1sSize(src.sizes());
  std::tie(b_src) = expand_inplace(dst, src.view(sliced_src_sizes), "setitem");
  dst.copy_(b_src);
}

static inline void recordTensorIndex(const Tensor& tensor, std::vector<Tensor>& outIndices, int64_t* dim_ptr) {
  // TODO: check scalarType
  outIndices.resize(*dim_ptr + 1);
  outIndices[*dim_ptr] = tensor;
  (*dim_ptr)++;
};

static inline Tensor handleSimpleTypesInSingleDimIndexingGet(
    const Tensor& self,
    const TensorIndex& index,
    bool is_tracing,
    const at::Device& self_device,
    const IntArrayRef& self_sizes) {
  if (index.is_integer()) {
    return applySelect(self, 0, index.integer(), 0, self_device, self_sizes);
  } else if (index.is_slice()) {
    return applySlice(
      self,
      0,
      index.slice().start(),
      index.slice().stop(),
      index.slice().step(),
      /*ensure_view=*/true,
      /*is_tracing=*/is_tracing,
      self_device,
      self_sizes);
  } else if (index.is_none()) {
    return self.unsqueeze(0);
  } else if (index.is_ellipsis()) {
    return at::alias(self);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Invalid TensorIndex type");
  }
}

static inline void handleSimpleTypesInSingleDimIndexingSet(
    const Tensor& self,
    const TensorIndex& index,
    const Tensor& value,
    bool is_tracing,
    const at::Device& self_device,
    const IntArrayRef& self_sizes) {
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

static inline Tensor handleDimInMultiDimIndexing(
    const Tensor& prev_dim_result,
    const Tensor& original_tensor,
    const TensorIndex& index,
    int64_t* dim_ptr,
    int64_t* specified_dims_ptr,
    int64_t real_dim,
    std::vector<Tensor>& outIndices,
    bool is_tracing,
    const at::Device& original_tensor_device,
    const IntArrayRef& prev_dim_result_sizes) {
  if (index.is_integer()) {
    return applySelect(prev_dim_result, *dim_ptr, index.integer(), real_dim, original_tensor_device, prev_dim_result_sizes);
  } else if (index.is_slice()) {
    Tensor result = applySlice(
      prev_dim_result,
      *dim_ptr,
      index.slice().start(),
      index.slice().stop(),
      index.slice().step(),
      /*ensure_view=*/false,
      /*is_tracing=*/is_tracing,
      original_tensor_device,
      prev_dim_result_sizes);
    (*dim_ptr)++;
    return result;
  } else if (index.is_ellipsis()) {
    (*dim_ptr) += original_tensor.dim() - (*specified_dims_ptr);
    return prev_dim_result;
  } else if (index.is_none()) {
    Tensor result = prev_dim_result.unsqueeze(*dim_ptr);
    (*dim_ptr)++;
    return result;
  } else if (index.is_boolean()) {
    Tensor result = prev_dim_result.unsqueeze(*dim_ptr);
    recordTensorIndex(boolToIndexingTensor(result, index.boolean(), original_tensor_device), outIndices, dim_ptr);
    return result;
  } else if (index.is_tensor()) {
    Tensor result = prev_dim_result;
    const Tensor& tensor = index.tensor();
    auto scalar_type = tensor.scalar_type();
    if (tensor.dim() == 0 && at::isIntegralType(scalar_type, /*includeBool=*/true)) {
      if (scalar_type != at::kByte && scalar_type != at::kBool) {
        result = applySelect(result, *dim_ptr, tensor.item<int64_t>(), real_dim, original_tensor_device, prev_dim_result_sizes);
      } else {
        result = result.unsqueeze(*dim_ptr);
        if (scalar_type == at::kBool) {
          recordTensorIndex(boolToIndexingTensor(result, tensor.item<bool>() != 0, original_tensor_device), outIndices, dim_ptr);
        } else {
          recordTensorIndex(boolToIndexingTensor(result, tensor.item<uint8_t>() != 0, original_tensor_device), outIndices, dim_ptr);
        }
      }
    } else {
      recordTensorIndex(tensor, outIndices, dim_ptr);
    }
    return result;
  } else {
    TORCH_INTERNAL_ASSERT(false, "Invalid TensorIndex type");
  }
}

static inline std::vector<Tensor> typeConvertIndices(const Tensor& self, std::vector<Tensor> indices) {
  std::vector<Tensor> converted_inds(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    const auto &ind = indices[i];
    if (ind.defined()) {
      converted_inds[i] = ind.to(ind.options().device(self.device()));
    } else {
      converted_inds[i] = std::move(indices[i]);
    }
  }
  return converted_inds;
}

static inline Tensor dispatch_index(const Tensor& self, std::vector<Tensor> indices) {
  OptionalDeviceGuard device_guard(device_of(self));
  return self.index(typeConvertIndices(self, std::move(indices)));
}

static inline Tensor dispatch_index_put_(Tensor& self, std::vector<Tensor> indices, const Tensor& value) {
  OptionalDeviceGuard device_guard(device_of(self));
  return self.index_put_(typeConvertIndices(self, std::move(indices)), value);
}

} // namespace indexing
} // namespace at
