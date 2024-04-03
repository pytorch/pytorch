#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Dispatch.h>
#include <ATen/native/sparse/Macros.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/SparseTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/result_type.h>
#endif

#ifdef GPUCC
#define NAME "sparse_binary_op_intersection_cuda"
#else
#define NAME "sparse_binary_op_intersection_cpu"
#endif

namespace at::native {

namespace {

using at::sparse::get_sparse_impl;

// ForwardIt: only legacy random access iterator is supported.
template<class ForwardIt, class T, bool is_lower = true>
static FUNCAPI INLINE
ForwardIt find_bound(ForwardIt first, ForwardIt last, const T& value) {
    ForwardIt RESTRICT it;
    typename std::iterator_traits<ForwardIt>::difference_type count, step;
    // NOTE: std::distance(first, last) compiles but produces wrong results on CUDA,
    // so only legacy random access iterators are safe in this code.
    count = last - first;

    while (count > 0) {
      it = first;
      step = count / 2;
      // avoiding std::advance(it, step),
      // although it does work unlike std::distance on CUDA.
      it += step;
      // The decision which separates finding a lower bound vs an upper bound.
      // Note that a lower bound is a value at *it with the smallest index
      // such that *it >= value if such value exists, or last if does not.
      // Similarly, an upper bound is a value at *it with the smallest index
      // such that *it > value if such value exists, or last if does not.
      // Let is_lower = true and *it < value, then we know that *it and values
      // preceeding *it cannot contain a lower bound, so we adjust initial iterator range
      // from [first, first + count] to [first + step + 1, first + count - (step + 1)],
      // where +1 skips the element at which we have just evaluated *it < value.
      // Samilar logic holds when is_lower = false.
      if (is_lower ? *it < value : value >= *it) {
        first = ++it;
        count -= step + 1;
      }
      else {
        count = step;
      }
    }
    return first;
}

template <template <typename func_t> class kernel_t>
struct KernelLauncher {
  template <typename func_t>
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    kernel_t<func_t>::launch(iter, f);
  }
};

TensorIterator make_value_selection_intersection_iter(
    const Tensor& lhs_values,
    const Tensor& lhs_select_idx,
    const Tensor& rhs_values,
    const Tensor& rhs_select_idx,
    const Tensor& intersection_counts) {
  const auto res_values_sizes = [&]() -> std::vector<int64_t> {
    auto sizes = infer_size(
        // keep nnz dim
        lhs_values.sizes(),
        // remove nnz dim for smooth broadcasting
        rhs_values.sizes().slice(1));
    // update nnz dim to be the length of an index
    sizes[0] = lhs_select_idx.numel();
    return sizes;
  }();
  auto res_values = at::empty(res_values_sizes, lhs_values.options());

  const auto restride_idx = [&res_values](const Tensor& idx) -> Tensor {
    auto idx_sizes = std::vector<int64_t>(res_values.dim(), 1);
    auto idx_strides = std::vector<int64_t>(res_values.dim(), 0);
    idx_sizes[0] = idx.numel();
    idx_strides[0] = 1;
    return idx.as_strided(idx_sizes, idx_strides);
  };

  const auto restride_values = [&lhs_select_idx](const Tensor& values) -> Tensor {
    auto values_sizes = at::DimVector(values.sizes());
    auto values_strides = at::DimVector(values.strides());
    values_sizes[0] = lhs_select_idx.numel();
    values_strides[0] = 0;
    return values.as_strided(values_sizes, values_strides);
  };

  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_owned_output(res_values)
    .add_owned_input(restride_values(lhs_values))
    .add_owned_input(restride_idx(lhs_select_idx))
    .add_owned_input(restride_values(rhs_values))
    .add_owned_input(restride_idx(rhs_select_idx))
    .add_owned_input(restride_idx(intersection_counts))
    .build();

  return iter;
}

template <
  template <typename func_t> class kernel_t,
  typename value_selection_intersection_kernel_t,
  typename index_t = int64_t,
  int64_t max_static_len = 0>
void _sparse_binary_op_intersection_kernel_impl(
    Tensor& res,
    const Tensor& x_,
    const Tensor& y_,
    const std::vector<int64_t>& broadcasted_shape,
    const c10::optional<Tensor>& x_hash_opt_ = c10::nullopt,
    const c10::optional<Tensor>& y_hash_opt_ = c10::nullopt,
    const bool accumulate_matches = true,
    const bool distributive_with_sum = true
) {
  // The common dtype check is relevant when op is done in-place.
  // This is because binary_of_t produces new values and it could be that
  // new_values.dtype != res.dtype. In such a case we should error out
  // as soon as possible to avoid redundant kernel runs.
  const auto common_dtype = at::result_type(x_, y_);
  TORCH_CHECK(canCast(common_dtype, res.scalar_type()),
      "Can't convert result type ", common_dtype,
      " to output ", res.scalar_type());

  using KernelLauncher = KernelLauncher<kernel_t>;
  using OptTensor = c10::optional<Tensor>;

  // If the op and sum are not distributive, coalesce is required.
  const auto coalesce_if_not_distributive = [distributive_with_sum](const Tensor& t, const OptTensor& t_hash_opt) -> auto {
    // No need to coalesce in such a case.
    if (distributive_with_sum) {
      return std::make_tuple(t, t_hash_opt);
    } else {
      // Otherwise coalesce and force hash recompute.
      return std::make_tuple(t.coalesce(), static_cast<OptTensor>(c10::nullopt));
    }
  };

  Tensor x, y;
  OptTensor x_hash_opt, y_hash_opt;
  std::tie(x, x_hash_opt) = coalesce_if_not_distributive(x_, x_hash_opt_);
  std::tie(y, y_hash_opt) = coalesce_if_not_distributive(y_, y_hash_opt_);

  // Given sparse tensors x and y we decide which one is source, and which one
  // is probably_coalesced. The indices of both source and probably_coalesced are
  // hashed and then the hash values of the source's indices are binary-searched
  // into the hash values of the probably_coalesced's indices.
  // If probably_coalesce is coalesced, by the property of the hashing method
  // (see below), the hash values are already sorted and we can avoid any
  // explicit sorting routines.
  Tensor probably_coalesced, source;
  OptTensor probably_coalesced_indices_hash_opt, source_indices_hash_opt;
  std::tie(probably_coalesced, probably_coalesced_indices_hash_opt, source, source_indices_hash_opt) = [&]() -> auto {
    // Case 1: either x or y is coalesced.
    if ((x.is_coalesced() ^ y.is_coalesced())) {
      return x.is_coalesced()
        ? std::make_tuple(x, x_hash_opt, y, y_hash_opt)
        : std::make_tuple(y, y_hash_opt, x, x_hash_opt);
    }
    // Case 2: Both x and y are either coalesced or non-coalesced.
    // If both are coalesced, search into the larger tensor is faster.
    // Same holds when both are non-coalesced.
    else {
      Tensor larger, smaller;
      OptTensor larger_hash_opt, smaller_hash_opt;
      std::tie(larger, larger_hash_opt, smaller, smaller_hash_opt) = [&]() -> auto {
        return x._nnz() >= y._nnz()
          ? std::make_tuple(x, x_hash_opt, y, y_hash_opt)
          : std::make_tuple(y, y_hash_opt, x, x_hash_opt);
      }();

      // If under a uniform distribution it is likely to hit many elements in larger,
      // it is best to coalesce it for better performance.
      const auto larger_sizes = larger.sizes();
      const auto sparse_dim_numel = std::accumulate(
          larger_sizes.begin(),
          larger_sizes.begin() + larger.sparse_dim(),
          1,
          std::multiplies<int64_t>());
      // If nnz > prod(larger.shape[:sparse_dim]), by the pidgeonhole principle,
      // there is at least one bucket with nnz / prod(larger.shape[:sparse_dim]) elements.
      // It provides a lower bound for the max count in the intersection.
      // This condition is very conservative as we do not check whether such an event
      // actually occurred, although it is very likely under a uniform distribution,
      // the distribution with the highest uncertainty (maximizes entropy).
      const auto max_count_lower_bound = larger._nnz() / sparse_dim_numel;
      constexpr int64_t MAX_COPIES_PER_THREAD = 50;
      return max_count_lower_bound > MAX_COPIES_PER_THREAD
        // coalesce invalidates hash values, so force-recompute
        ? std::make_tuple(larger.coalesce(), static_cast<OptTensor>(c10::nullopt), smaller, smaller_hash_opt)
        : std::make_tuple(larger, larger_hash_opt, smaller, smaller_hash_opt);
    }
  }();

  // The employed hash function maps a d-dim index to a linear offset
  // into a contiguous memory that is sufficient to fit a dense tensor
  // of shape broadcasted_shape(x.shape, y.shape), i.e.
  // idx -> \sum_{i = 0}^d idx[i] * hash_coeffs[i], where
  // hash_coeffs are the strides of a contiguous tensor of shape
  // broadcasted_shape(x.shape, y.shape).
  // Assuming the following order on the dimensions, i.e. the right-most dim is the
  // fastest-changing dim, and the left-most is the slowest-changing dim,
  // which is implicit in the definition of hash_coeffs,
  // it could be shown that the hash function is actually bijective and, hence,
  // is a perfect hash function (no collisions ever).

  // Need owning storage in case of the Tensor class.
  const auto hash_coeffs_storage = [&]() -> auto {
    const auto broadcasted_sparse_dim_shape = std::vector<int64_t>(
      broadcasted_shape.begin(),
      broadcasted_shape.begin() + probably_coalesced.sparse_dim()
    );
    auto strides = c10::contiguous_strides(broadcasted_sparse_dim_shape);
    return at::sparse::TensorGeometryHolder<max_static_len>(strides, strides, probably_coalesced.options());
  }();

  const auto hash_coeffs = std::get<0>(*hash_coeffs_storage);

  const auto nnz_arange = at::arange(
      std::max(probably_coalesced._nnz(), source._nnz()),
      source._indices().options());
  const auto probably_coalesced_nnz_arange = nnz_arange.narrow(-1, 0, probably_coalesced._nnz());

  // non-const because of gcc-5/clang-5 issues
  auto sparse_dim = probably_coalesced.sparse_dim();

  // Apply the hash function to probably_coalesced.indices
  const auto probably_coalesced_indices_hash = [&]() -> Tensor {
    // probably_coalesced is coalesced and hash provided? Reuse it!
    if (probably_coalesced_indices_hash_opt.has_value()) {
      return (*probably_coalesced_indices_hash_opt).contiguous();
    }

    const auto indices = probably_coalesced._indices();
    // non-const because of gcc-5/clang-5 issues
    auto indices_dim_stride = indices.stride(0);
    auto indices_nnz_stride = indices.stride(1);

    auto hash = at::empty({probably_coalesced._nnz()}, indices.options().dtype(kLong));

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .add_output(hash)
      .add_input(probably_coalesced_nnz_arange)
      .build();

    {
      const auto* RESTRICT ptr_indices = indices.data_ptr<index_t>();

      KernelLauncher::launch(iter,
          // NOTE: capture by value required by CUDA
          [=] FUNCAPI (index_t nnz_idx) -> int64_t {
          const auto* RESTRICT ptr_indices_dim = ptr_indices ? ptr_indices + nnz_idx * indices_nnz_stride : nullptr;
          int64_t hash = 0;
          for (int64_t dim = 0; dim < sparse_dim; ++dim) {
            const auto dim_hash_coeff = hash_coeffs[dim];
            const auto dim_index = ptr_indices_dim[dim * indices_dim_stride];
            hash += dim_index * dim_hash_coeff;
          }
          return hash;
      });
    }

    return hash;
  }();

  // Now that we have hash values of probably_coalesced.indices,
  // we need to decide whether they need to get sorted.
  // The sort is not requires if probably_coalesced is coalesced.
  Tensor sorted_hash, argsort_hash;
  std::tie(sorted_hash, argsort_hash) = [&]() -> std::tuple<Tensor, Tensor> {
    if (probably_coalesced.is_coalesced()) {
      // NOTE: argsort.dtype == nnz_arange.dtype
      const auto argsort = nnz_arange.narrow(-1, 0, probably_coalesced._nnz());
      return std::make_tuple(probably_coalesced_indices_hash, argsort);
    } else {
      // NOTE: we want argsort.dtype == nnz_arange.dtype,
      // but sort() produces indices of type int64_t,
      // so we convert to nnz_arange.dtype to avoid issues
      // with pointer types in the kernels below.
      Tensor sorted, argsort;
      std::tie(sorted, argsort) = probably_coalesced_indices_hash.sort();
      return std::make_tuple(sorted, argsort.to(nnz_arange.scalar_type()));
    }
  }();

  // Perform hash intersection.
  // Let  s_hash = hash(source.indices),
  //     pc_hash = hash(probably_coalesced.indices), then
  // for i = 0, ..., len(s_hash) - 1:
  //     lb = <index of a value in pc_hash[argsort_hash] which is a lower bound for s_hash[i]>,
  //     up = <index of a value in pc_hash[argsort_hash] which is an upper bound for s_hash[i]>,
  //     intersection_count[i] = up - lb
  //     intersection_first_idx[i] = lb.
  //
  // intersection_count and intersection_first_idx are used to form indices at which
  // intersection values are selected.
  Tensor intersection_count, intersection_first_idx;
  std::tie(intersection_count, intersection_first_idx) = [&]() -> std::tuple<Tensor, Tensor> {
    const auto source_nnz = source._nnz();
    auto intersection_buffer = at::empty({2, source_nnz}, sorted_hash.options());
    auto intersection_count = intersection_buffer.select(0, 0);
    auto intersection_first_idx = intersection_buffer.select(0, 1);

    const auto source_indices = source._indices();
    const auto source_arange = nnz_arange.narrow(-1, 0, source_nnz);
    // non-const because of gcc-5/clang-5 issues
    auto indices_dim_stride = source_indices.stride(0);
    auto indices_nnz_stride = source_indices.stride(1);
    auto dummy = at::empty({1}, source_arange.options());

    auto hash = source_indices_hash_opt.has_value()
      ? (*source_indices_hash_opt).contiguous()
      : at::empty({0}, probably_coalesced._indices().options().dtype(kLong));
    const auto* RESTRICT hash_ptr = source_indices_hash_opt.has_value()
      ? hash.data_ptr<int64_t>()
      : nullptr;

    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .add_owned_output(dummy.expand_as(source_arange))
      .add_input(source_arange)
      .build();

    {
      const auto* RESTRICT ptr_indices = source_indices.data_ptr<index_t>();
      const auto* RESTRICT ptr_sorted_hash = sorted_hash.data_ptr<int64_t>();
      const auto sorted_hash_len = sorted_hash.numel();
      auto* RESTRICT ptr_intersection_count = intersection_count.data_ptr<int64_t>();
      auto* RESTRICT ptr_intersection_first_idx = intersection_first_idx.data_ptr<int64_t>();

      // Fusing hash computation with hash intersection.
      KernelLauncher::launch(iter,
          // NOTE: capture by value required by CUDA
          [=] FUNCAPI (index_t nnz_idx) -> index_t {
          int64_t hash = 0;
          if (hash_ptr) {
            hash = hash_ptr[nnz_idx];
          } else if (sparse_dim) {
            // Compute hash value
            const auto* RESTRICT ptr_indices_dim = ptr_indices + nnz_idx * indices_nnz_stride;
            for (int64_t dim = 0; dim < sparse_dim; ++dim) {
              const auto dim_hash_coeff = hash_coeffs[dim];
              const auto dim_index = ptr_indices_dim[dim * indices_dim_stride];
              hash += dim_index * dim_hash_coeff;
            }
          }

          // Perform hash values intersection
          const auto* RESTRICT lb = find_bound<const int64_t*, int64_t, /*is_lower=*/true>(
              ptr_sorted_hash,
              ptr_sorted_hash + sorted_hash_len,
              hash
          );

          const auto* RESTRICT ub = find_bound<const int64_t*, int64_t, /*is_lower=*/false>(
              ptr_sorted_hash,
              ptr_sorted_hash + sorted_hash_len,
              hash
          );

          ptr_intersection_count[nnz_idx] = ub - lb;
          ptr_intersection_first_idx[nnz_idx] = lb - ptr_sorted_hash;

          return 0;
      });
    }

    return std::make_tuple(intersection_count, intersection_first_idx);
  }();

  const auto res_indices = source._indices().clone();
  const auto binary_op_res_dtype = at::result_type(source._values(), probably_coalesced._values());
  const auto res_values = value_selection_intersection_kernel_t::apply(
      source._values().to(binary_op_res_dtype),
      nnz_arange.narrow(-1, 0, source._nnz()),
      probably_coalesced._values().to(binary_op_res_dtype),
      intersection_first_idx.to(nnz_arange.scalar_type()),
      intersection_count,
      argsort_hash,
      accumulate_matches).to(res.scalar_type());
  const auto res_sparse_dim = source.sparse_dim();
  const auto res_dense_dim = source.dense_dim();
  const auto& res_shape = broadcasted_shape;
  const auto res_nnz = source._nnz();

  auto* res_sparse_impl = get_sparse_impl(res);
  res_sparse_impl->raw_resize_(res_sparse_dim, res_dense_dim, res_shape);
  res_sparse_impl->set_indices_and_values_unsafe(res_indices, res_values);
  res_sparse_impl->set_nnz_and_narrow(res_nnz);
  res._coalesced_(source.is_coalesced());
}

template <
  template <typename func_t> class kernel_t,
  typename value_selection_intersection_kernel_t>
void _sparse_binary_op_intersection_kernel_out(
    Tensor& res,
    const Tensor& x,
    const Tensor& y,
    const c10::optional<Tensor>& x_hash_opt = c10::nullopt,
    const c10::optional<Tensor>& y_hash_opt = c10::nullopt,
    // If op distributes with the sum, the arguments are processed as is,
    // without the calls to coalesce().
    const bool distributive_with_sum = true
) {
  TORCH_CHECK(
      (x.is_sparse() && y.is_sparse())
      && (x.dim() == y.dim()) && (x.sparse_dim() == y.sparse_dim())
      && (x.sizes().slice(0, x.sparse_dim()) == y.sizes().slice(0, y.sparse_dim())),
      NAME, "(): expects sparse inputs with equal dimensionality, ",
      "number of sparse dimensions, and shape of sparse dimensions");
  TORCH_CHECK(
      x._indices().scalar_type() == y._indices().scalar_type(),
      NAME, "(): expects inputs' indices to be of the same dtype (i.e. long or int)");

  const auto check_hash_validity = [](const Tensor& t, const c10::optional<Tensor>& t_hash_opt) {
    if (!t_hash_opt.has_value()) {
      return;
    }

    const auto &t_hash = *t_hash_opt;
    TORCH_INTERNAL_ASSERT(
        t_hash.dim() == 1 && t_hash.scalar_type() == kLong && t_hash.size(-1) == t._indices().size(-1),
        NAME, "(): explicit hash values need to be a 1-dim Long tensor with the ",
        "NSE matching that of the corresponding sparse tensor.");
  };

  check_hash_validity(x, x_hash_opt);
  check_hash_validity(y, y_hash_opt);

  const auto broadcasted_shape = infer_size(x.sizes(), y.sizes());

  // 8 sparse dims should be more than enough?
  constexpr int64_t max_sparse_dims = 8;

  // COO indices are only 64-bit integers for now.
  using index_t = int64_t;

  if (max_sparse_dims > x.sparse_dim()) {
    _sparse_binary_op_intersection_kernel_impl<
      // For some reason MSVC complaints about passing constexpr max_sparse_dims
      // as a template parameter claiming as if it is not know at compile time.
      kernel_t, value_selection_intersection_kernel_t, index_t, 8>(
        res, x, y, broadcasted_shape, x_hash_opt, y_hash_opt, distributive_with_sum);
  } else {
    _sparse_binary_op_intersection_kernel_impl<
      kernel_t, value_selection_intersection_kernel_t, index_t>(
        res, x, y, broadcasted_shape, x_hash_opt, y_hash_opt, distributive_with_sum);
  }
}

} // anonymous namespace

} // at::native
