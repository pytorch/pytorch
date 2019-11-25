#pragma once

#include <c10/util/Optional.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/DeviceGuard.h>

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
  Slice();
  Slice(int64_t start, int64_t stop, int64_t step);

  int64_t start() const;
  int64_t stop() const;
  int64_t step() const;

 private:
  int64_t start_;
  int64_t stop_;
  int64_t step_;
};

CAFFE2_API std::ostream& operator<<(std::ostream& stream, const Slice& slice);

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
  TensorIndex(c10::nullopt_t);

  // Case 2: "..." / `at::indexing::Ellipsis`
  TensorIndex(at::indexing::EllipsisIndexType);
  TensorIndex(const char *str);

  // Case 3: Integer value
  TensorIndex(int64_t integer);
  TensorIndex(int integer);

  // Case 4: Boolean value
  template <class T,
            class = typename std::enable_if<std::is_same<bool, T>::value>::type >
  TensorIndex(T boolean) : boolean_(boolean), type_(TensorIndexType::Boolean) {}

  // Case 5: Slice represented in `{start, stop, step}` form,
  // where `start` / `stop` / `step` can be integer or `at::indexing::None`
  TensorIndex(std::initializer_list<c10::optional<int64_t>> slice);

  // Case 5: Tensor value
  TensorIndex(Tensor tensor);

  bool is_none() const;
  bool is_ellipsis() const;

  bool is_integer() const;
  int64_t integer() const;

  bool is_boolean() const;
  bool boolean() const;

  bool is_slice() const;
  const Slice& slice() const;

  bool is_tensor() const;
  const Tensor& tensor() const;

 private:
  int64_t integer_;
  bool boolean_;
  Slice slice_;
  Tensor tensor_;
  TensorIndexType type_;
};

CAFFE2_API std::ostream& operator<<(std::ostream& stream, const TensorIndex& tensor_index);
CAFFE2_API std::ostream& operator<<(std::ostream& stream, const std::vector<TensorIndex>& tensor_indices);

inline Tensor applySlice(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t stop,
    int64_t step,
    bool ensure_view=false,
    bool is_tracing=false) {
  const auto& length = self.size(dim);

  TORCH_CHECK_VALUE(step != 0, "step cannot be zero");
  // TODO: implement negative step
  TORCH_CHECK_VALUE(step >= 0, "negative step not yet supported");

  // Skip this optimization if we are tracing, as the trace may be polymorphic
  // over the shape of the `self` tensor, and we still want to record
  // the slice.
  if (!ensure_view && start == 0 && stop == length && step == 1 && !is_tracing) {
    return self;
  }
  return self.slice(dim, start, stop, step);
}

inline Tensor applySelect(const Tensor& self, int64_t dim, int64_t index, int64_t real_dim=0) {
  TORCH_CHECK_INDEX(
    !(index == 0 && dim == 0 && self.dim() == 0),
    "invalid index of a 0-dim tensor. ",
    "Use tensor.item() to convert a 0-dim tensor to a number");

  int64_t size = self.size(dim);
  TORCH_CHECK_INDEX(
    index >= -size && index < size,
    "index ", index, " is out of bounds for dimension ", real_dim, " with size ", size);

  // if the index is negative, do not normalize it because that would fix the index
  // on the current tensor size in the tracer.
  // aten::select also works on negative indices
  return self.select(dim, index);
}

inline Tensor boolToIndexingTensor(const Tensor& self, bool value) {
  // booleans add a dimension of size 1. true indexes this dimension as if 0:, false as empty.
  if (value) {
    return at::native::zeros({1}, {}, self.options().dtype(kLong));
  } else {
    return at::native::empty({0}, {}, self.options().dtype(kLong));
  }
}

inline void recordTensorIndex(const Tensor& tensor, std::vector<Tensor>& outIndices, int64_t& dim) {
  // TODO: check scalarType
  outIndices.resize(dim + 1);
  outIndices[dim] = tensor;
  dim++;
};

inline Tensor handleInteger(const Tensor& self, int64_t& dim, int64_t index, int64_t real_dim) {
  return applySelect(self, dim, index, real_dim);
}

inline Tensor handleSlice(const Tensor& self, int64_t& dim, int64_t start, int64_t stop, int64_t step, bool is_tracing=false) {
  Tensor result = applySlice(self, dim, start, stop, step, /*ensure_view=*/false, /*is_tracing=*/is_tracing);
  dim++;
  return result;
}

inline void handleEllipsis(const Tensor& self, int64_t& dim, int64_t specified_dims) {
  dim += self.dim() - specified_dims;
}

inline Tensor handleNone(const Tensor& self, int64_t& dim) {
  Tensor result = self.unsqueeze(dim);
  dim++;
  return result;
}

inline Tensor handleBoolean(const Tensor& self, bool boolean, std::vector<Tensor>& outIndices, int64_t& dim) {
  Tensor result = self.unsqueeze(dim);
  recordTensorIndex(boolToIndexingTensor(result, boolean), outIndices, dim);
  return result;
}

inline Tensor handleTensor(const Tensor& self, const Tensor& tensor, std::vector<Tensor>& outIndices, int64_t& dim, int64_t real_dim) {
  Tensor result = self;
  auto scalar_type = tensor.scalar_type();
  if (tensor.dim() == 0 && at::isIntegralType(scalar_type, /*includeBool=*/true)) {
    if (scalar_type != at::kByte && scalar_type != at::kBool) {
      result = applySelect(result, dim, tensor.item<int64_t>(), real_dim);
    } else {
      result = result.unsqueeze(dim);
      if (scalar_type == at::kBool) {
        recordTensorIndex(boolToIndexingTensor(result, tensor.item<bool>() != 0), outIndices, dim);
      } else {
        recordTensorIndex(boolToIndexingTensor(result, tensor.item<uint8_t>() != 0), outIndices, dim);
      }
    }
  } else {
    recordTensorIndex(tensor, outIndices, dim);
  }
  return result;
}

inline Tensor handleNoneSingleDim(const Tensor& self) {
  return self.unsqueeze(0);
}

inline Tensor handleEllipsisSingleDim(const Tensor& self) {
  return self.alias();
}

inline Tensor handleIntegerSingleDim(const Tensor& self, int64_t index) {
  return applySelect(self, 0, index);
}

inline Tensor handleSliceSingleDim(const Tensor& self, int64_t start, int64_t stop, int64_t step, bool is_get, bool is_tracing=false) {
  return applySlice(self, 0, start, stop, step, /*ensure_view=*/is_get, /*is_tracing=*/is_tracing);
}

inline std::vector<Tensor> typeConvertIndices(const Tensor& self, const std::vector<Tensor>& indices) {
  std::vector<Tensor> converted_inds(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    const auto &ind = indices[i];
    if (ind.defined()) {
      converted_inds[i] = ind.to(ind.options().device(self.device()));
    } else {
      converted_inds[i] = indices[i];
    }
  }
  return converted_inds;
}

inline Tensor dispatch_index(const Tensor& self, const std::vector<Tensor>& indices) {
  std::vector<Tensor> converted_indices = typeConvertIndices(self, indices);
  OptionalDeviceGuard device_guard(device_of(self));
  return self.index(converted_indices);
}

inline Tensor dispatch_index_put_(Tensor& self, const std::vector<Tensor>& indices, const Tensor& value) {
  std::vector<Tensor> converted_indices = typeConvertIndices(self, indices);
  OptionalDeviceGuard device_guard(device_of(self));
  return self.index_put_(converted_indices, value);
}

// To match numpy semantics:
// As a special case for backwards compatibility,
// strip away unit dimensions from the left of 'src'
inline IntArrayRef slicePrefix1sSize(IntArrayRef sizes) {
  size_t first_non1_src = sizes.size();
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] != 1) {
      first_non1_src = i;
      break;
    }
  }

  return sizes.slice(first_non1_src);
}

inline void copy_to(Tensor dst, const Tensor& src) {
  Tensor b_src;
  IntArrayRef sliced_src_sizes = slicePrefix1sSize(src.sizes());
  std::tie(b_src) = expand_inplace(dst, src.view(sliced_src_sizes), "setitem");
  dst.copy_(b_src);
}

} // namespace indexing
} // namespace at
