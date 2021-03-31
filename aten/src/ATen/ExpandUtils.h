#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/MaybeOwned.h>

#include <functional>
#include <sstream>
#include <tuple>

namespace at {

TORCH_API std::vector<int64_t> infer_size(IntArrayRef a, IntArrayRef b);
TORCH_API std::tuple<std::vector<int64_t>, std::vector<int64_t>>
inferExpandGeometry(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes);

TORCH_API std::vector<int64_t> infer_dense_strides(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides);

// True if input shapes are expandable
// NOTE: infer_size did a similar check, please keep them sync if change is needed
inline bool are_expandable(IntArrayRef shape1, IntArrayRef shape2) {
  size_t ndim1 = shape1.size();
  size_t ndim2 = shape2.size();
  size_t ndim = ndim1 < ndim2 ? ndim1 : ndim2;

  for (int64_t i = ndim - 1; i >= 0; --i) {
    if (shape1[--ndim1] == shape2[--ndim2] || shape1[ndim1] == 1 || shape2[ndim2] == 1) {
      continue;
    }
    return false;
  }
  return true;
}

// avoid copy-construction of Tensor by using a reference_wrapper.
inline void check_defined(std::initializer_list<std::reference_wrapper<const Tensor>> tensors, const char *api_name) {
  for (auto& t : tensors) {
    if (!t.get().defined()) {
      AT_ERROR(api_name, "(...) called with an undefined Tensor");
    }
  }
}

inline c10::MaybeOwned<Tensor> expand_inplace_v2(const Tensor& tensor, const Tensor& to_expand) {
  if (tensor.sizes().equals(to_expand.sizes())) {
    return c10::MaybeOwned<Tensor>::borrowed(to_expand);
  }
  return c10::MaybeOwned<Tensor>::owned(to_expand.expand(tensor.sizes(), /*implicit=*/true)); // see [expand implicit]
}

// expand_inplace_v2 may borrow from to_expand; prevent calling with an
// obvious temporary.
inline c10::MaybeOwned<Tensor> expand_inplace_v2(const Tensor& tensor, Tensor&& to_expand) = delete;


C10_DEPRECATED_MESSAGE("expand_inplace is deprecated. Use expand_inplace_v2 instead.")
inline std::tuple<Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand) {
  return std::make_tuple(*expand_inplace_v2(tensor, to_expand));
}

inline c10::MaybeOwned<Tensor> expand_inplace_v2(const Tensor &tensor, const Tensor &to_expand, const char *api_name) {
  check_defined({tensor, to_expand}, api_name);
  return expand_inplace_v2(tensor, to_expand);
}

// expand_inplace_v2 may borrow from to_expand; prevent calling with an
// obvious temporary.
inline c10::MaybeOwned<Tensor> expand_inplace_v2(const Tensor& tensor, Tensor&& to_expand, const char *api_name) = delete;


C10_DEPRECATED_MESSAGE("expand_inplace is deprecated. Use expand_inplace_v2 instead.")
inline std::tuple<Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand, const char *api_name) {
  return std::make_tuple(*expand_inplace_v2(tensor, to_expand, api_name));
}

inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>> expand_inplace_v2(const Tensor &tensor, const Tensor &to_expand1, const Tensor &to_expand2) {
  if (tensor.sizes().equals(to_expand1.sizes()) && tensor.sizes().equals((to_expand2.sizes()))) {
    return std::make_tuple(
        c10::MaybeOwned<Tensor>::borrowed(to_expand1),
        c10::MaybeOwned<Tensor>::borrowed(to_expand2));
  }

  return std::make_tuple(
      c10::MaybeOwned<Tensor>::owned(to_expand1.expand(tensor.sizes(), /*implicit=*/true)), // see [expand implicit]
      c10::MaybeOwned<Tensor>::owned(to_expand2.expand(tensor.sizes(), /*implicit=*/true)));
}

// expand_inplace_v2 may borrow from to_expand1 and to_expand2; prevent
// calling with an obvious temporary.
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>> expand_inplace_v2(const Tensor &tensor, Tensor &&to_expand1, const Tensor &to_expand2) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>> expand_inplace_v2(const Tensor &tensor, const Tensor &to_expand1, Tensor &&to_expand2) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>> expand_inplace_v2(const Tensor &tensor, Tensor &&to_expand1, Tensor &&to_expand2) = delete;


C10_DEPRECATED_MESSAGE("expand_inplace is deprecated. Use expand_inplace_v2 instead.")
inline std::tuple<Tensor, Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand1, const Tensor &to_expand2) {
  auto newResult = expand_inplace_v2(tensor, to_expand1, to_expand2);
  return std::make_tuple(*std::get<0>(newResult), *std::get<1>(newResult));
}

inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>> expand_inplace_v2(const Tensor &tensor, const Tensor &to_expand1, const Tensor &to_expand2,
                                                 const char *api_name) {
  check_defined({tensor, to_expand1, to_expand2}, api_name);
  return expand_inplace_v2(tensor, to_expand1, to_expand2);
}

// expand_inplace_v2 may borrow from to_expand1 and to_expand2; prevent
// calling with an obvious temporary.
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>> expand_inplace_v2(const Tensor &tensor, Tensor &&to_expand1, const Tensor &to_expand2, const char *api_name) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>> expand_inplace_v2(const Tensor &tensor, const Tensor &to_expand1, Tensor &&to_expand2, const char *api_name) = delete;
inline std::tuple<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>> expand_inplace_v2(const Tensor &tensor, Tensor &&to_expand1, Tensor &&to_expand2, const char *api_name) = delete;


C10_DEPRECATED_MESSAGE("expand_inplace is deprecated. Use expand_inplace_v2 instead.")
inline std::tuple<Tensor, Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand1, const Tensor &to_expand2,
                                                 const char *api_name) {
  auto newResult = expand_inplace_v2(tensor, to_expand1, to_expand2, api_name);
  return std::make_tuple(*std::get<0>(newResult), *std::get<1>(newResult));
}

inline std::tuple<Tensor, Tensor> expand_outplace(const Tensor &to_expand1, const Tensor &to_expand2) {
  if (to_expand1.sizes().equals(to_expand2.sizes())) {
    return std::make_tuple(to_expand1, to_expand2);
  }

  auto expanded_size = infer_size(to_expand1.sizes(), to_expand2.sizes());
  return std::make_tuple(
      to_expand1.expand(expanded_size, /*implicit=*/true), // see [expand implicit]
      to_expand2.expand(expanded_size, /*implicit=*/true));
}

inline std::tuple<Tensor, Tensor> expand_outplace(const Tensor &to_expand1, const Tensor &to_expand2, const char *api_name) {
  check_defined({to_expand1, to_expand2}, api_name);
  return expand_outplace(to_expand1, to_expand2);
}

inline std::tuple<Tensor, Tensor, Tensor> expand_outplace(const Tensor &to_expand1,
                                                          const Tensor &to_expand2,
                                                          const Tensor &to_expand3) {
  if (to_expand1.sizes().equals(to_expand2.sizes()) && to_expand1.sizes().equals(to_expand3.sizes())) {
    return std::make_tuple(to_expand1, to_expand2, to_expand3);
  }

  auto expanded_size12 = infer_size(to_expand1.sizes(), to_expand2.sizes());
  auto expanded_size = infer_size(expanded_size12, to_expand3.sizes());
  return std::make_tuple(
      to_expand1.expand(expanded_size, /*implicit=*/true), // see [expand implicit]
      to_expand2.expand(expanded_size, /*implicit=*/true),
      to_expand3.expand(expanded_size, /*implicit=*/true));
}

inline std::tuple<Tensor, Tensor, Tensor> expand_outplace(const Tensor &to_expand1,
                                                          const Tensor &to_expand2,
                                                          const Tensor &to_expand3,
                                                          const char *api_name) {
  check_defined({to_expand1, to_expand2, to_expand3}, api_name);
  return expand_outplace(to_expand1, to_expand2, to_expand3);
}

inline std::tuple<Tensor> expand_size(const Tensor &to_expand, IntArrayRef sizes) {
  if(to_expand.sizes().equals(sizes)) {
    return std::make_tuple(to_expand);
  }

  return std::make_tuple(to_expand.expand(sizes, /*implicit=*/true)); // see [expand implicit]
}

inline std::tuple<Tensor> expand_size(const Tensor &to_expand, IntArrayRef sizes, const char *api_name) {
  check_defined({to_expand}, api_name);
  return expand_size(to_expand, sizes);
}

inline std::vector<Tensor> expand_outplace(TensorList to_expand) {
  // expands a list of Tensors; ignores undefined (null) tensors
  bool first = true;
  std::vector<int64_t> sizes;
  for (size_t i = 0; i < to_expand.size(); ++i) {
    if (!to_expand[i].defined()) {
      continue;
    } else if (first) {
      sizes = to_expand[i].sizes().vec();
      first = false;
    } else {
      sizes = infer_size(sizes, to_expand[i].sizes());
    }
  }

  std::vector<Tensor> result(to_expand.size());
  for (size_t i = 0; i < to_expand.size(); ++i) {
    if (!to_expand[i].defined()) {
      continue;
    } else if (to_expand[i].sizes().equals(sizes)) {
      result[i] = to_expand[i];
    } else {
      result[i] = to_expand[i].expand(sizes, /*implicit=*/true); // see [expand implicit]
    }
  }
  return result;
}

// Sums `tensor` repeatedly to produce a tensor of shape `shape`.
// Precondition: is_expandable_to(shape, tensor.sizes()) must be true
static inline Tensor sum_to(Tensor tensor, const IntArrayRef shape) {
  if (shape.size() == 0) {
    return tensor.sum();
  }
  c10::SmallVector<int64_t, 8> reduce_dims;
  const at::IntArrayRef sizes = tensor.sizes();
  const int64_t leading_dims = sizes.size() - shape.size();
  for (int64_t i = 0; i < leading_dims; ++i) {
    reduce_dims.push_back(i);
  }
  for (int64_t i = leading_dims; i < static_cast<int64_t>(sizes.size()); ++i) {
    if (shape[i - leading_dims] == 1 && sizes[i] != 1) {
      reduce_dims.push_back(i);
    }
  }
  if (!reduce_dims.empty()) {
    tensor = tensor.sum(reduce_dims, /*keepdim=*/true);
  }
  return leading_dims > 0 ? tensor.view(shape) : tensor;
}

// True if `shape` can be broadcasted to `desired`
static inline bool is_expandable_to(IntArrayRef shape, IntArrayRef desired) {
  size_t ndim = shape.size();
  size_t target_dim = desired.size();
  if (ndim > target_dim) {
    return false;
  }
  for (size_t i = 0; i < ndim; i++) {
    int64_t size = shape[ndim - i - 1];
    int64_t target = desired[target_dim - i - 1];
    if (size != target && size != 1) {
      return false;
    }
  }
  return true;
}

}
