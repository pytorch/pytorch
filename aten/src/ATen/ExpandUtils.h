#pragma once

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/view.h>
#include <ATen/ops/view_copy.h>
#endif

#include <ATen/Tensor.h>
#include <ATen/core/DimVector.h>
#include <c10/util/Exception.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/irange.h>

#include <functional>
#include <tuple>
#include <utility>

namespace at {

TORCH_API std::vector<int64_t> infer_size(IntArrayRef a, IntArrayRef b);
TORCH_API std::vector<SymInt> infer_size_symint(
    SymIntArrayRef a,
    SymIntArrayRef b);
TORCH_API DimVector infer_size_dimvector(IntArrayRef a, IntArrayRef b);
TORCH_API SymDimVector
infer_size_symdimvector(SymIntArrayRef a, SymIntArrayRef b);

// Named type instead of a pair/tuple so that we can be sure to
// construct the vectors in place and get NRVO.
template <typename Container>
struct InferExpandGeometryResult {
  Container sizes;
  Container strides;
  explicit InferExpandGeometryResult(size_t ndim)
      : sizes(ndim), strides(ndim) {}
  explicit InferExpandGeometryResult(IntArrayRef sizes_, size_t ndim)
      : sizes(sizes_.begin(), sizes_.end()), strides(ndim) {}
};

TORCH_API std::tuple<std::vector<int64_t>, std::vector<int64_t>>
inferExpandGeometry(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes);

TORCH_API InferExpandGeometryResult<DimVector> inferExpandGeometry_dimvector(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes);

TORCH_API std::vector<int64_t> infer_dense_strides(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides);

// True if input shapes are expandable
// NOTE: infer_size did a similar check, please keep them sync if change is
// needed
inline bool are_expandable(IntArrayRef shape1, IntArrayRef shape2) {
  size_t ndim1 = shape1.size();
  size_t ndim2 = shape2.size();
  size_t ndim = ndim1 < ndim2 ? ndim1 : ndim2;

  for (int64_t i = static_cast<int64_t>(ndim) - 1; i >= 0; --i) {
    if (shape1[--ndim1] == shape2[--ndim2] || shape1[ndim1] == 1 ||
        shape2[ndim2] == 1) {
      continue;
    }
    return false;
  }
  return true;
}

// avoid copy-construction of Tensor by using a reference_wrapper.
inline void check_defined(
    std::initializer_list<std::reference_wrapper<const Tensor>> tensors,
    const char* api_name) {
  for (auto& t : tensors) {
    if (!t.get().defined()) {
      TORCH_CHECK(false, api_name, "(...) called with an undefined Tensor");
    }
  }
}

// NOTE [ ExpandUtils Borrowing ]
//
// Functions in ExpandUtils return `c10::MaybeOwned<Tensor>` because
// expansion may not actually be needed, in which case we can improve
// efficiency by returning
// `c10::MaybeOwned<Tensor>::borrowed(to_expand)`. However, this means
// that you need to be careful: the returned `c10::MaybeOwned<Tensor>`
// must not outlive the original `Tensor` object that `to_expand`
// referred to! The deleted rvalue reference overloads of these
// functions help with this by preventing trivial use of a temporary
// resulting from a function call, but it is still possible to make a
// mistake.

inline c10::MaybeOwned<Tensor> expand_inplace(
    const Tensor& tensor,
    const Tensor& to_expand) {
  if (tensor.sym_sizes().equals(to_expand.sym_sizes())) {
    return c10::MaybeOwned<Tensor>::borrowed(to_expand);
  }
  return c10::MaybeOwned<Tensor>::owned(
      to_expand.expand_symint(tensor.sym_sizes()));
}

inline c10::MaybeOwned<Tensor> expand_inplace(
    const Tensor& tensor,
    Tensor&& to_expand) = delete;

inline c10::MaybeOwned<Tensor> expand_inplace(
    const Tensor& tensor,
    const Tensor& to_expand,
    const char* api_name) {
  check_defined({tensor, to_expand}, api_name);
  return expand_inplace(tensor, to_expand);
}

inline c10::MaybeOwned<Tensor> expand_inplace(
    const Tensor& tensor,
    Tensor&& to_expand,
    const char* api_name) = delete;

inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(
    const Tensor& tensor,
    const Tensor& to_expand1,
    const Tensor& to_expand2) {
  if (tensor.sizes().equals(to_expand1.sizes()) &&
      tensor.sizes().equals((to_expand2.sizes()))) {
    return std::make_tuple(
        c10::MaybeOwned<Tensor>::borrowed(to_expand1),
        c10::MaybeOwned<Tensor>::borrowed(to_expand2));
  }

  return std::make_tuple(
      c10::MaybeOwned<Tensor>::owned(to_expand1.expand(tensor.sizes())),
      c10::MaybeOwned<Tensor>::owned(to_expand2.expand(tensor.sizes())));
}

inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(
    const Tensor& tensor,
    Tensor&& to_expand1,
    const Tensor& to_expand2) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(
    const Tensor& tensor,
    const Tensor& to_expand1,
    Tensor&& to_expand2) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(const Tensor& tensor, Tensor&& to_expand1, Tensor&& to_expand2) =
    delete;

inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(
    const Tensor& tensor,
    const Tensor& to_expand1,
    const Tensor& to_expand2,
    const char* api_name) {
  check_defined({tensor, to_expand1, to_expand2}, api_name);
  return expand_inplace(tensor, to_expand1, to_expand2);
}

inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(
    const Tensor& tensor,
    Tensor&& to_expand1,
    const Tensor& to_expand2,
    const char* api_name) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(
    const Tensor& tensor,
    const Tensor& to_expand1,
    Tensor&& to_expand2,
    const char* api_name) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_inplace(
    const Tensor& tensor,
    Tensor&& to_expand1,
    Tensor&& to_expand2,
    const char* api_name) = delete;

// See NOTE [ ExpandUtils Borrowing ] above for `MaybeOwned` explanation.
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(const Tensor& to_expand1, const Tensor& to_expand2) {
  auto s1 = to_expand1.sym_sizes();
  auto s2 = to_expand2.sym_sizes();
  if (s1.equals(s2)) {
    return std::make_tuple(
        c10::MaybeOwned<Tensor>::borrowed(to_expand1),
        c10::MaybeOwned<Tensor>::borrowed(to_expand2));
  }

  auto expanded_size = infer_size_symdimvector(s1, s2);
  return std::make_tuple(
      c10::MaybeOwned<Tensor>::owned(to_expand1.expand_symint(expanded_size)),
      c10::MaybeOwned<Tensor>::owned(to_expand2.expand_symint(expanded_size)));
}

inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(Tensor&& to_expand1, const Tensor& to_expand2) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(const Tensor& to_expand1, Tensor&& to_expand2) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(Tensor&& to_expand1, Tensor&& to_expand2) = delete;

inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    const Tensor& to_expand2,
    const char* api_name) {
  check_defined({to_expand1, to_expand2}, api_name);
  return expand_outplace(to_expand1, to_expand2);
}

inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    const Tensor& to_expand2,
    const char* api_name) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    Tensor&& to_expand2,
    const char* api_name) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    Tensor&& to_expand2,
    const char* api_name) = delete;

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    const Tensor& to_expand2,
    const Tensor& to_expand3) {
  if (to_expand1.sizes().equals(to_expand2.sizes()) &&
      to_expand1.sizes().equals(to_expand3.sizes())) {
    return std::make_tuple(
        c10::MaybeOwned<Tensor>::borrowed(to_expand1),
        c10::MaybeOwned<Tensor>::borrowed(to_expand2),
        c10::MaybeOwned<Tensor>::borrowed(to_expand3));
  }

  auto expanded_size12 =
      infer_size_dimvector(to_expand1.sizes(), to_expand2.sizes());
  auto expanded_size =
      infer_size_dimvector(expanded_size12, to_expand3.sizes());
  return std::make_tuple(
      c10::MaybeOwned<Tensor>::owned(to_expand1.expand(expanded_size)),
      c10::MaybeOwned<Tensor>::owned(to_expand2.expand(expanded_size)),
      c10::MaybeOwned<Tensor>::owned(to_expand3.expand(expanded_size)));
}

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    const Tensor& to_expand2,
    const Tensor& to_expand3) = delete;
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    Tensor&& to_expand2,
    const Tensor& to_expand3) = delete;
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    Tensor&& to_expand2,
    const Tensor& to_expand3) = delete;
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    const Tensor& to_expand2,
    Tensor&& to_expand3) = delete;
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    const Tensor& to_expand2,
    Tensor&& to_expand3) = delete;
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    Tensor&& to_expand2,
    Tensor&& to_expand3) = delete;
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(Tensor&& to_expand1, Tensor&& to_expand2, Tensor&& to_expand3) =
    delete;

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    const Tensor& to_expand2,
    const Tensor& to_expand3,
    const char* api_name) {
  check_defined({to_expand1, to_expand2, to_expand3}, api_name);
  return expand_outplace(to_expand1, to_expand2, to_expand3);
}

inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    const Tensor& to_expand2,
    const Tensor& to_expand3,
    const char* api_name) = delete;
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    Tensor&& to_expand2,
    const Tensor& to_expand3,
    const char* api_name) = delete;
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    Tensor&& to_expand2,
    const Tensor& to_expand3,
    const char* api_name) = delete;
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    const Tensor& to_expand2,
    Tensor&& to_expand3,
    const char* api_name) = delete;
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    const Tensor& to_expand2,
    Tensor&& to_expand3,
    const char* api_name) = delete;
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    const Tensor& to_expand1,
    Tensor&& to_expand2,
    Tensor&& to_expand3,
    const char* api_name) = delete;
inline std::tuple<
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>,
    c10::MaybeOwned<Tensor>>
expand_outplace(
    Tensor&& to_expand1,
    Tensor&& to_expand2,
    Tensor&& to_expand3,
    const char* api_name) = delete;

inline c10::MaybeOwned<Tensor> expand_size(
    const Tensor& to_expand,
    IntArrayRef sizes) {
  if (to_expand.sizes().equals(sizes)) {
    return c10::MaybeOwned<Tensor>::borrowed(to_expand);
  }

  return c10::MaybeOwned<Tensor>::owned(to_expand.expand(sizes));
}

inline c10::MaybeOwned<Tensor> expand_size(
    Tensor&& to_expand,
    IntArrayRef sizes) = delete;

inline c10::MaybeOwned<Tensor> expand_size(
    const Tensor& to_expand,
    IntArrayRef sizes,
    const char* api_name) {
  check_defined({to_expand}, api_name);
  return expand_size(to_expand, sizes);
}

inline c10::MaybeOwned<Tensor> expand_size(
    Tensor&& to_expand,
    IntArrayRef sizes,
    const char* api_name) = delete;

inline std::vector<Tensor> expand_outplace(TensorList to_expand) {
  // expands a list of Tensors; ignores undefined (null) tensors
  bool first = true;
  SymDimVector sizes;
  for (const auto i : c10::irange(to_expand.size())) {
    if (!to_expand[i].defined()) {
      continue;
    } else if (first) {
      sizes = to_expand[i].sym_sizes();
      first = false;
    } else {
      sizes = infer_size_symdimvector(sizes, to_expand[i].sym_sizes());
    }
  }

  std::vector<Tensor> result(to_expand.size());
  for (const auto i : c10::irange(to_expand.size())) {
    if (!to_expand[i].defined()) {
      continue;
    } else if (to_expand[i].sym_sizes().equals(sizes)) {
      result[i] = to_expand[i];
    } else {
      result[i] = to_expand[i].expand_symint(sizes);
    }
  }
  return result;
}

template <typename T>
inline Tensor _sum_to(
    Tensor tensor,
    const c10::ArrayRef<T> shape,
    bool always_return_non_view = false) {
  if (shape.size() == 0) {
    return tensor.sum();
  }

  auto sizes = at::symint::sizes<T>(tensor);
  c10::SmallVector<int64_t, 8> reduce_dims;
  const int64_t leading_dims = sizes.size() - shape.size();
  for (const auto i : c10::irange(leading_dims)) {
    reduce_dims.push_back(i);
  }
  for (int64_t i = leading_dims; i < static_cast<int64_t>(sizes.size()); ++i) {
    if (TORCH_GUARD_OR_FALSE(sym_eq(shape[i - leading_dims], 1)) &&
        TORCH_GUARD_OR_TRUE(sym_ne(sizes[i], 1))) {
      reduce_dims.push_back(i);
    } else {
      // if we assume no reduction due to unbacked we ensure that at runtime.
      TORCH_MAYBE_SYM_CHECK(
          sym_eq(shape[i - leading_dims], sizes[i]),
          "non-reduction path was assumed due to unabcked symbols expected those two sizes to be the same:",
          shape[i - leading_dims],
          ", ",
          sizes[i])
    }
  }

  if (!reduce_dims.empty()) {
    tensor = tensor.sum(reduce_dims, /*keepdim=*/true);
  }

  if (always_return_non_view) {
    // This is only actually used by the functionalization pass.
    // We want to be able to guarantee that this function doesn't return a view
    // of the input.
    return leading_dims > 0 ? at::symint::view_copy<T>(tensor, shape)
                            : tensor.clone();
  } else {
    return leading_dims > 0 ? at::symint::view<T>(tensor, shape) : tensor;
  }
}

inline Tensor sum_to(
    Tensor tensor,
    const c10::SymIntArrayRef shape,
    bool always_return_non_view = false) {
  return _sum_to(std::move(tensor), shape, always_return_non_view);
}

// Sums `tensor` repeatedly to produce a tensor of shape `shape`.
// Precondition: is_expandable_to(shape, tensor.sizes()) must be true
inline Tensor sum_to(
    Tensor tensor,
    const IntArrayRef shape,
    bool always_return_non_view = false) {
  return _sum_to(std::move(tensor), shape, always_return_non_view);
}

inline bool is_expandable_to(
    SymIntArrayRef shape,
    c10::SymIntArrayRef desired) {
  size_t ndim = shape.size();
  size_t target_dim = desired.size();
  if (ndim > target_dim) {
    return false;
  }
  for (const auto i : c10::irange(ndim)) {
    const auto& size = shape[ndim - i - 1];
    const auto& target = desired[target_dim - i - 1];
    if (size != target && size != 1) {
      return false;
    }
  }
  return true;
}

inline bool is_expandable_to(IntArrayRef shape, IntArrayRef desired) {
  auto sym_shape = c10::SymIntArrayRef(
      reinterpret_cast<const c10::SymInt*>(shape.data()), shape.size());
  auto sym_desired = c10::SymIntArrayRef(
      reinterpret_cast<const c10::SymInt*>(desired.data()), desired.size());
  return is_expandable_to(sym_shape, sym_desired);
}

} // namespace at
