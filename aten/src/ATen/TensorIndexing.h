#pragma once

#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/SymInt.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/alias.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/core/List.h>

#include <utility>

namespace at {
namespace indexing {

const int64_t INDEX_MIN = c10::SymInt::min_representable_int();
const int64_t INDEX_MAX = -(INDEX_MIN + 1);

enum class TensorIndexType { None, Ellipsis, SymInt, Boolean, Slice, Tensor };

constexpr c10::nullopt_t None = c10::nullopt;

struct TORCH_API EllipsisIndexType final {
  EllipsisIndexType() = default;
};
TORCH_API extern const EllipsisIndexType Ellipsis;

struct TORCH_API Slice final {
 public:
  Slice(
      c10::optional<c10::SymInt> start_index = c10::nullopt,
      c10::optional<c10::SymInt> stop_index = c10::nullopt,
      c10::optional<c10::SymInt> step_index = c10::nullopt) {
    if (!step_index.has_value()) {
      step_ = c10::SymInt(1);
    } else {
      step_ = std::move(step_index).value();
    }

    TORCH_CHECK_VALUE(step_ != 0, "slice step cannot be zero");

    if (!start_index.has_value()) {
      start_ = c10::SymInt(step_ < 0 ? INDEX_MAX : 0);
    } else {
      start_ = std::move(start_index).value();
    }

    if (!stop_index.has_value()) {
      stop_ = c10::SymInt(step_ < 0 ? INDEX_MIN : INDEX_MAX);
    } else {
      stop_ = std::move(stop_index).value();
    }
  }

  inline c10::SymInt start() const {
    return start_;
  }

  inline c10::SymInt stop() const {
    return stop_;
  }

  inline c10::SymInt step() const {
    return step_;
  }

 private:
  c10::SymInt start_;
  c10::SymInt stop_;
  c10::SymInt step_;
};

TORCH_API std::ostream& operator<<(std::ostream& stream, const Slice& slice);

// `at::indexing::TensorIndex` is used for converting C++ tensor indices such as
// `{None, "...", Ellipsis, 0, true, Slice(1, None, 2), torch::tensor({1, 2})}`
// into its equivalent `std::vector<TensorIndex>`, so that further tensor
// indexing operations can be performed using the supplied indices.
//
// There is one-to-one correspondence between Python and C++ tensor index types:
// Python                  | C++
// -----------------------------------------------------
// `None`                  | `at::indexing::None`
// `Ellipsis`              | `at::indexing::Ellipsis`
// `...`                   | `"..."`
// `123`                   | `123`
// `True` / `False`        | `true` / `false`
// `:`                     | `Slice()` / `Slice(None, None)`
// `::`                    | `Slice()` / `Slice(None, None, None)`
// `1:`                    | `Slice(1, None)`
// `1::`                   | `Slice(1, None, None)`
// `:3`                    | `Slice(None, 3)`
// `:3:`                   | `Slice(None, 3, None)`
// `::2`                   | `Slice(None, None, 2)`
// `1:3`                   | `Slice(1, 3)`
// `1::2`                  | `Slice(1, None, 2)`
// `:3:2`                  | `Slice(None, 3, 2)`
// `1:3:2`                 | `Slice(1, 3, 2)`
// `torch.tensor([1, 2])`) | `torch::tensor({1, 2})`
struct TORCH_API TensorIndex final {
  // Case 1: `at::indexing::None`
  TensorIndex(c10::nullopt_t) : type_(TensorIndexType::None) {}

  // Case 2: "..." / `at::indexing::Ellipsis`
  TensorIndex(at::indexing::EllipsisIndexType)
      : type_(TensorIndexType::Ellipsis) {}
  TensorIndex(const char* str) : TensorIndex(at::indexing::Ellipsis) {
    TORCH_CHECK_VALUE(
        strcmp(str, "...") == 0,
        "Expected \"...\" to represent an ellipsis index, but got \"",
        str,
        "\"");
  }

  // Case 3: (Sym) Integer value
  TensorIndex(SymInt integer)
      : integer_(std::move(integer)), type_(TensorIndexType::SymInt) {}
  TensorIndex(int64_t integer) : TensorIndex(SymInt(integer)) {}
  TensorIndex(int integer) : TensorIndex(SymInt(integer)) {}

  // Case 4: Boolean value
  template <
      class T,
      class = typename std::enable_if<std::is_same<bool, T>::value>::type>
  TensorIndex(T boolean) : boolean_(boolean), type_(TensorIndexType::Boolean) {}

  // Case 5: Slice represented in `at::indexing::Slice` form
  TensorIndex(Slice slice)
      : slice_(std::move(slice)), type_(TensorIndexType::Slice) {}

  // Case 6: Tensor value
  TensorIndex(Tensor tensor)
      : tensor_(std::move(tensor)), type_(TensorIndexType::Tensor) {}

  inline bool is_none() const {
    return type_ == TensorIndexType::None;
  }

  inline bool is_ellipsis() const {
    return type_ == TensorIndexType::Ellipsis;
  }

  inline bool is_integer() const {
    return type_ == TensorIndexType::SymInt;
  }

  inline SymInt integer() const {
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
  SymInt integer_ = 0;
  bool boolean_ = false;
  Slice slice_;
  Tensor tensor_;
  TensorIndexType type_;
};

TORCH_API std::ostream& operator<<(
    std::ostream& stream,
    const TensorIndex& tensor_index);
TORCH_API std::ostream& operator<<(
    std::ostream& stream,
    const std::vector<TensorIndex>& tensor_indices);

namespace impl {
static inline Tensor applySlice(
    const Tensor& self,
    int64_t dim,
    c10::SymInt start,
    c10::SymInt stop,
    c10::SymInt step,
    bool disable_slice_optimization,
    const at::Device& self_device,
    const c10::optional<SymIntArrayRef>& self_sizes) {
  // TODO: implement negative step
  TORCH_CHECK_VALUE(step > 0, "step must be greater than zero");

  // See NOTE [nested tensor size for indexing]
  if (self_sizes.has_value()) {
    // Skip this optimization if we are tracing, as the trace may be polymorphic
    // over the shape of the `self` tensor, and we still want to record
    // the slice.
    SymInt length = (self_device == at::kCPU || self_device == at::kCUDA)
        ? (*self_sizes)[dim]
        : self.sym_size(dim);
    if (!disable_slice_optimization && start == 0 && length == stop &&
        step == 1) {
      return self;
    }
  }
  return self.slice_symint(dim, start, stop, std::move(step));
}

static inline Tensor applySelect(
    const Tensor& self,
    int64_t dim,
    SymInt index,
    int64_t real_dim,
    const at::Device& /*self_device*/,
    const c10::optional<SymIntArrayRef>& self_sizes) {
  // See NOTE [nested tensor size for indexing]
  if (self_sizes.has_value()) {
    auto maybe_index = index.maybe_as_int();
    if (maybe_index.has_value()) {
      TORCH_CHECK_INDEX(
          !(maybe_index.value() == 0 && dim == 0 && self_sizes->empty()),
          "invalid index of a 0-dim tensor. ",
          "Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number");
    }

    auto size = (*self_sizes)[dim];
    TORCH_CHECK_INDEX(
        size >= -index && size > index,
        "index ",
        index,
        " is out of bounds for dimension ",
        real_dim,
        " with size ",
        size);
  }

  // if the index is negative, do not normalize it because that would fix the
  // index on the current tensor size in the tracer. aten::select also works on
  // negative indices
  return self.select_symint(dim, index);
}

static inline Tensor boolToIndexingTensorCPUOrCUDA(
    const Tensor& self,
    bool value) {
  // booleans add a dimension of size 1. true indexes this dimension as if 0:,
  // false as empty.
  if (value) {
    return at::empty({1}, self.options().dtype(kLong)).fill_(0.);
  } else {
    return at::empty({0}, self.options().dtype(kLong));
  }
}

static inline Tensor boolToIndexingTensorNonNativeDeviceType(
    const Tensor& self,
    bool value) {
  // booleans add a dimension of size 1. true indexes this dimension as if 0:,
  // false as empty.
  if (value) {
    return at::zeros({1}, self.options().dtype(kLong));
  } else {
    return at::empty({0}, self.options().dtype(kLong));
  }
}

static inline Tensor boolToIndexingTensor(
    const Tensor& self,
    bool value,
    const at::Device& self_device) {
  if (self_device == at::kCPU || self_device == at::kCUDA) {
    return boolToIndexingTensorCPUOrCUDA(self, value);
  } else {
    return boolToIndexingTensorNonNativeDeviceType(self, value);
  }
}

static inline Tensor scalarToTensorNonNativeDeviceType(
    const Scalar& v,
    const TensorOptions& options) {
  return at::scalar_tensor(v, options);
}

static inline void recordTensorIndex(
    const Tensor& tensor,
    std::vector<Tensor>& outIndices,
    int64_t* dim_ptr) {
  // TODO: check scalarType
  outIndices.resize(*dim_ptr + 1);
  outIndices[*dim_ptr] = tensor;
  (*dim_ptr)++;
};

static inline c10::List<c10::optional<Tensor>> typeConvertIndices(
    const Tensor& /*self*/,
    std::vector<Tensor>&& indices) {
  c10::List<c10::optional<Tensor>> converted_inds;
  converted_inds.reserve(indices.size());
  for (const auto& i : indices) {
    converted_inds.push_back(std::move(i));
  }
  return converted_inds;
}

// NOTE: Why do we mirror instead of replace the `count_specified_dimensions`
// function in torch/csrc/autograd/python_variable_indexing.cpp? It's because
// `count_specified_dimensions` is on the hot path of Python tensor multi-dim
// indexing (i.e. it's called by `applySlicing` which is called by
// `THPVariable_getitem` / `THPVariable_setitem` when handling indexing of more
// than one dimension). If we were to merge the Python/C++
// `count_specified_dimensions` function, on the Python side we would have to
// construct a `std::vector` container to be consumed by the C++
// `count_specified_dimensions` function, which adds 100s of nanoseconds
// overhead and is undesirable.
static inline int64_t count_specified_dimensions(
    const ArrayRef<TensorIndex>& indices) {
  // Count the number of indexed dimensions (everything but ellipsis and None)
  int64_t count = 0;
  for (auto& obj : indices) {
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
} // namespace impl

// NOTE: Many functions below are only for consumption from Python indexing
// implementation, they include:
//
// - `Tensor scalarToTensor(...)`
// - `IntArrayRef slicePrefix1sSize(...)`
// - `void copy_to(...)`
// - `Tensor handleDimInMultiDimIndexing(...)`
// - `Tensor dispatch_index(...)`
// - `Tensor dispatch_index_put_(...)`
// - `Tensor get_item(...)`
// - `void set_item(...)`
//
// The rest of the functions are in `at::indexing::impl` namespace, signifying
// that they shouldn't be used from Python indexing implementation.
static inline Tensor scalarToTensor(
    const Scalar& v,
    const TensorOptions& options,
    const at::Device& self_device) {
  if (self_device == at::kCPU) {
    return at::detail::scalar_tensor_static(
        v, options.dtype_opt()->toScalarType(), self_device);
  } else {
    return impl::scalarToTensorNonNativeDeviceType(v, options);
  }
}

// To match numpy semantics:
// As a special case for backwards compatibility,
// strip away unit dimensions from the left of 'src'
static inline SymIntArrayRef slicePrefix1sSize(const SymIntArrayRef& sizes) {
  size_t first_non1_src = sizes.size();
  for (const auto i : c10::irange(sizes.size())) {
    // Unbacked SymInt has different behavior, but this is sound because
    // failing to slice will only ever cause an error, not divergent
    // behavior
    if (!sizes[i].has_hint() || sizes[i] != 1) {
      first_non1_src = i;
      break;
    }
  }

  return sizes.slice(first_non1_src);
}

static inline void copy_to(const Tensor& dst, const Tensor& src) {
  if (dst.sym_sizes().equals(src.sym_sizes())) {
    // A shortcut to avoid generating hard-coded constant sizes during tracing.
    // This is not a perfect solution: when src & dst have different shapes,
    // constants will still appear. Users can workaround that case by
    // dst[index..] = src.reshape(..)
    dst.copy_(src);
    return;
  } else if (src.dim() == 0 && src.device().type() == at::kCPU) {
    dst.fill_(src);
    return;
  }
  auto src_view = src.view_symint(slicePrefix1sSize(src.sym_sizes()));
  c10::MaybeOwned<Tensor> b_src = expand_inplace(dst, src_view, "setitem");
  dst.copy_(*b_src);
}

// See NOTE [ Setting `disable_slice_optimization` when calling C++ tensor
// indexing functions from Python ]
static inline Tensor handleDimInMultiDimIndexing(
    const Tensor& prev_dim_result,
    const Tensor& original_tensor,
    const TensorIndex& index,
    int64_t* dim_ptr,
    int64_t* specified_dims_ptr,
    int64_t real_dim,
    std::vector<Tensor>& outIndices,
    bool disable_slice_optimization,
    const at::Device& original_tensor_device,
    const c10::optional<SymIntArrayRef>& prev_dim_result_sizes) {
  if (index.is_integer()) {
    return impl::applySelect(
        prev_dim_result,
        *dim_ptr,
        index.integer(),
        real_dim,
        original_tensor_device,
        prev_dim_result_sizes);
  } else if (index.is_slice()) {
    Tensor result = impl::applySlice(
        prev_dim_result,
        *dim_ptr,
        index.slice().start(),
        index.slice().stop(),
        index.slice().step(),
        /*disable_slice_optimization=*/disable_slice_optimization,
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
    impl::recordTensorIndex(
        impl::boolToIndexingTensor(
            result, index.boolean(), original_tensor_device),
        outIndices,
        dim_ptr);
    return result;
  } else if (index.is_tensor()) {
    Tensor result = prev_dim_result;
    const Tensor& tensor = index.tensor();
    auto scalar_type = tensor.scalar_type();
    if (tensor.dim() == 0 &&
        at::isIntegralType(scalar_type, /*includeBool=*/true)) {
      if (scalar_type != at::kByte && scalar_type != at::kBool) {
        result = impl::applySelect(
            result,
            *dim_ptr,
            tensor.item<int64_t>(),
            real_dim,
            original_tensor_device,
            prev_dim_result_sizes);
      } else {
        result = result.unsqueeze(*dim_ptr);
        if (scalar_type == at::kBool) {
          impl::recordTensorIndex(
              impl::boolToIndexingTensor(
                  result, tensor.item<bool>() != 0, original_tensor_device),
              outIndices,
              dim_ptr);
        } else {
          impl::recordTensorIndex(
              impl::boolToIndexingTensor(
                  result, tensor.item<uint8_t>() != 0, original_tensor_device),
              outIndices,
              dim_ptr);
        }
      }
    } else {
      impl::recordTensorIndex(tensor, outIndices, dim_ptr);
    }
    return result;
  } else {
    TORCH_INTERNAL_ASSERT(false, "Invalid TensorIndex type");
  }
}

namespace impl {
// This mirrors `applySlicing` in
// torch/csrc/autograd/python_variable_indexing.cpp
static inline Tensor applySlicing(
    const Tensor& self,
    const ArrayRef<TensorIndex>& indices,
    std::vector<Tensor>& outIndices,
    bool disable_slice_optimization,
    const at::Device& self_device,
    const c10::optional<SymIntArrayRef>& self_sizes) {
  int64_t dim = 0;
  int64_t specified_dims = impl::count_specified_dimensions(indices);

  // See NOTE [nested tensor size for indexing]
  if (self_sizes.has_value()) {
    TORCH_CHECK_INDEX(
        specified_dims <= (int64_t)self_sizes->size(),
        "too many indices for tensor of dimension ",
        (int)self_sizes->size());
  }

  Tensor result = self;
  for (const auto i : c10::irange(indices.size())) {
    auto& obj = indices[i];
    // See NOTE [nested tensor size for indexing]
    c10::optional<SymIntArrayRef> result_sizes = result.is_nested()
        ? c10::optional<SymIntArrayRef>(c10::nullopt)
        : c10::optional<SymIntArrayRef>(result.sym_sizes());
    result = handleDimInMultiDimIndexing(
        /*prev_dim_result=*/result,
        /*original_tensor=*/self,
        /*index=*/obj,
        /*dim=*/&dim,
        /*specified_dims=*/&specified_dims,
        /*real_dim=*/i,
        /*outIndices=*/outIndices,
        /*disable_slice_optimization=*/disable_slice_optimization,
        /*original_tensor_device=*/self_device,
        /*prev_dim_result_sizes=*/result_sizes);
  }
  return result;
}
} // namespace impl

static inline Tensor dispatch_index(
    const Tensor& self,
    std::vector<Tensor>&& indices) {
  return self.index(impl::typeConvertIndices(self, std::move(indices)));
}

static inline Tensor dispatch_index_put_(
    Tensor& self,
    std::vector<Tensor>&& indices,
    const Tensor& value) {
  return self.index_put_(
      impl::typeConvertIndices(self, std::move(indices)), value);
}

// NOTE [ Setting `disable_slice_optimization` when calling C++ tensor indexing
// functions from Python ]
//
// Question: When should we set `disable_slice_optimization` to `true` when
// calling C++ tensor indexing functions from Python indexing code?
//
// Answer: What "slice optimization" means: when we have a slicing expression
// like `x[0:5, 0]`, where the sliced tensor was of size 5 in dimension 0, we
// would skip dispatching the actual slice call as an optimization. However,
// here are the cases where we DON'T want this optimization:
//
// 1. When we are doing 1-D slicing (e.g. `tensor[:]`).
//    Reason: we always return a shallow copy for expressions such as
//    `tensor[:]` / `tensor[...]` / `tensor[:, :]`. (Note that for `tensor[:,
//    :]`, we return an alias of `tensor` by doing the following:
//    ```
//    Tensor sliced = impl::applySlicing(self, indices, tensorIndices,
//    disable_slice_optimization, self_device, self_sizes); if
//    (tensorIndices.empty()) {
//      if (sliced.is_same(self)) {
//        // ensure we return a shallow copy for things like x[...]
//        sliced = at::alias(sliced);
//      }
//      return sliced;
//    }
//    ```)
// 2. When we are doing JIT tracing.
//    Reason: JIT tracing needs the `self.slice(...)` call to properly trace the
//    slice operation.

// This mirrors `THPVariable_getitem` in
// torch/csrc/autograd/python_variable_indexing.cpp See NOTE [ Setting
// `disable_slice_optimization` when calling C++ tensor indexing functions from
// Python ]
static inline Tensor get_item(
    const Tensor& self,
    const ArrayRef<TensorIndex>& indices,
    bool disable_slice_optimization = false) {
  at::Device self_device = self.device();
  // NOTE [nested tensor size for indexing]
  // nested tensor does not have a size (yet) so for now we represent its size
  // as null may need to be changed after we reach a better solution for nested
  // tensor size
  c10::optional<SymIntArrayRef> self_sizes = self.is_nested()
      ? c10::optional<SymIntArrayRef>(c10::nullopt)
      : c10::optional<SymIntArrayRef>(self.sym_sizes());

  // handle simple types: integers, slices, none, ellipsis, bool
  if (indices.size() == 1) {
    const TensorIndex& index = indices[0];
    if (index.is_integer()) {
      return impl::applySelect(
          self, 0, index.integer(), 0, self_device, self_sizes);
    } else if (index.is_slice()) {
      return impl::applySlice(
          self,
          0,
          index.slice().start(),
          index.slice().stop(),
          index.slice().step(),
          /*disable_slice_optimization=*/true,
          self_device,
          self_sizes);
    } else if (index.is_none()) {
      return self.unsqueeze(0);
    } else if (index.is_ellipsis()) {
      return at::alias(self);
    } else if (index.is_boolean()) {
      Tensor result = self.unsqueeze(0);
      return dispatch_index(
          result,
          std::vector<Tensor>{impl::boolToIndexingTensor(
              result, index.boolean(), self_device)});
    }
  }

  std::vector<Tensor> tensorIndices;
  Tensor sliced = impl::applySlicing(
      self,
      indices,
      tensorIndices,
      disable_slice_optimization,
      self_device,
      self_sizes);
  if (tensorIndices.empty()) {
    if (sliced.is_same(self)) {
      // ensure we return a shallow copy for things like x[...]
      sliced = at::alias(sliced);
    }
    return sliced;
  }

  // indexing by tensors ("advanced" indexing)
  return dispatch_index(sliced, std::move(tensorIndices));
}

// This mirrors `THPVariable_setitem` in
// torch/csrc/autograd/python_variable_indexing.cpp for "the assigned value is a
// Tensor" case See NOTE [ Setting `disable_slice_optimization` when calling C++
// tensor indexing functions from Python ]
static inline void set_item(
    const Tensor& self,
    const ArrayRef<TensorIndex>& indices,
    const Tensor& value,
    bool disable_slice_optimization = false) {
  at::Device self_device = self.device();
  SymIntArrayRef self_sizes = self.sym_sizes();

  // handle simple types: integers, slices, ellipsis, bool
  if (indices.size() == 1) {
    const TensorIndex& index = indices[0];
    if (index.is_boolean() && !index.boolean()) {
      // do nothing for false (technically we should check the size, but we
      // don't have real 0-sized shapes.
      return;
    } else if (index.is_ellipsis()) {
      copy_to(self, value);
      return;
    } else if (index.is_none() || (index.is_boolean() && index.boolean())) {
      copy_to(self.unsqueeze(0), value);
      return;
    } else if (index.is_integer()) {
      copy_to(
          impl::applySelect(
              self, 0, index.integer(), 0, self_device, self_sizes),
          value);
      return;
    } else if (index.is_slice()) {
      copy_to(
          impl::applySlice(
              self,
              0,
              index.slice().start(),
              index.slice().stop(),
              index.slice().step(),
              /*disable_slice_optimization=*/disable_slice_optimization,
              self_device,
              self_sizes),
          value);
      return;
    }
  }

  std::vector<Tensor> tensorIndices;
  Tensor sliced = impl::applySlicing(
      self,
      indices,
      tensorIndices,
      disable_slice_optimization,
      self_device,
      self_sizes);
  if (tensorIndices.empty()) {
    copy_to(sliced, value);
    return;
  }

  SymIntArrayRef valueSizes = value.sym_sizes();
  SymIntArrayRef slicedValueSizes = slicePrefix1sSize(valueSizes);
  Tensor valuesSliced;
  if (!valueSizes.equals(slicedValueSizes)) {
    valuesSliced = value.view_symint(slicedValueSizes);
  } else {
    valuesSliced = value;
  }
  dispatch_index_put_(sliced, std::move(tensorIndices), valuesSliced);
  return;
}

} // namespace indexing
} // namespace at
