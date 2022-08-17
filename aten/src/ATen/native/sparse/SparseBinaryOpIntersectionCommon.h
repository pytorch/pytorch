#pragma once

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Dispatch.h>
#include <ATen/native/sparse/Macros.h>
#include <ATen/ExpandUtils.h>
#include <ATen/SparseTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/result_type.h>
#endif

#ifdef GPUCC
#define NAME "sparse_binary_op_intersection_cuda"
#else
#define NAME "sparse_binary_op_intersection_cpu"
#endif

#define BINARY_OP_API static constexpr FUNCAPI INLINE

namespace at {
namespace native {

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

template <
  template <typename func_t> class kernel_t,
  typename binary_op_t,
  typename hash_t = int64_t,
  typename offset_t = int64_t>
Tensor& _sparse_binary_op_intersection_kernel_impl(
    Tensor& res,
    const Tensor& x_,
    const Tensor& y_,
    const std::vector<int64_t> broadcasted_shape,
    const bool is_commutative = true
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

  const Tensor x = is_commutative ? x_ : x_.coalesce();
  const Tensor y = is_commutative ? y_ : y_.coalesce();

  Tensor probably_coalesced, source;
  std::tie(probably_coalesced, source) = [&]() -> std::tuple<Tensor, Tensor> {
    // Case 1: either x or y is coalesced.
    if ((x.is_coalesced() ^ y.is_coalesced())) {
      return x.is_coalesced()
        ? std::make_tuple(x, y)
        : std::make_tuple(y, x);
    }
    // Case 2: Both x and y are either coalesced or non-coalesced.
    // If both are coalesced, search into the larger tensor is faster.
    // Same holds when both are non-coalesced.
    else {
      return x._nnz() >= y._nnz()
        ? std::make_tuple(x, y)
        : std::make_tuple(y, x);
    }
  }();

  const auto kHash = std::is_same<hash_t, int64_t>::value ? kLong : kInt;
  const auto hash_coeffs = [&]() -> Tensor {
    const auto broadcasted_sparse_dim_shape = std::vector<int64_t> {
      broadcasted_shape.begin(),
      broadcasted_shape.begin() + probably_coalesced.sparse_dim()
    };
    auto strides = contiguous_strides(broadcasted_sparse_dim_shape);
    const auto hash_coeffs_cpu = at::from_blob(
        reinterpret_cast<void*>(strides.data()),
        {static_cast<int64_t>(strides.size())},
        probably_coalesced._indices().options().device(kCPU).dtype(kLong))
      .to(kHash);
    return hash_coeffs_cpu.to(probably_coalesced.device());
  }();

  const auto nnz_arange = at::arange(
      std::max(probably_coalesced._nnz(), source._nnz()),
      source._indices().options());
  const auto probably_coalesced_nnz_arange = nnz_arange.narrow(-1, 0, probably_coalesced._nnz());

  // non-const because of gcc-5/clang-5 issues
  auto sparse_dim = probably_coalesced.sparse_dim();
  const auto probably_coalesced_indices_hash = [&]() -> Tensor {
    const auto indices = probably_coalesced._indices();
    // non-const because of gcc-5/clang-5 issues
    auto indices_dim_stride = indices.stride(0);
    auto indices_nnz_stride = indices.stride(1);

    auto hash = at::empty({probably_coalesced._nnz()},
        indices.options().dtype(kHash));

    auto iter = TensorIteratorConfig()
      // Hash has hash_t type while probably_coalesced_nnz_arange is index_t.
      .check_all_same_dtype(false)
      .add_output(hash)
      .add_input(probably_coalesced_nnz_arange)
      .build();

    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), NAME, [&]() {
        const auto* RESTRICT ptr_indices = indices.data_ptr<index_t>();
        const auto* RESTRICT ptr_hash_coeffs = hash_coeffs.template data_ptr<hash_t>();
        // non-const because of gcc-5/clang-5 issues
        auto sdim = static_cast<uint32_t>(sparse_dim);

        KernelLauncher::launch(iter,
            // Windows does not seem to like these nested captures without explicit names.
            [ptr_indices, indices_dim_stride, indices_nnz_stride, sdim, ptr_hash_coeffs]
            FUNCAPI (index_t nnz_idx) -> hash_t {
            const auto* RESTRICT ptr_indices_dim = ptr_indices + nnz_idx * indices_nnz_stride;
            auto hash = hash_t {0};
            for (uint32_t dim = 0; dim < sdim; ++dim) {
              // use only int32_t operations when hash_t == int32_t
              const auto dim_hash_coeff = ptr_hash_coeffs[dim];
              const auto dim_index = static_cast<hash_t>(ptr_indices_dim[dim * indices_dim_stride]);
              hash += dim_index * dim_hash_coeff;
            }
            return hash;
        });
    });

    return hash;
  }();

  Tensor sorted_hash, argsort_hash;
  std::tie(sorted_hash, argsort_hash) = [&]() -> std::tuple<Tensor, Tensor> {
    if (probably_coalesced.is_coalesced()) {
      // NOTE: argsort.dtype == nnz_arange.dtype
      const auto argsort = nnz_arange.narrow(-1, 0, probably_coalesced._nnz());
      return std::make_tuple(probably_coalesced_indices_hash, argsort);
    }
    else {
      // NOTE: we want argsort.dtype == nnz_arange.dtype,
      // but sort() produces indices of type int64_t,
      // so we convert to nnz_arange.dtype to avoid issues
      // with pointer types in the kernels below.
      Tensor sorted, argsort;
      std::tie(sorted, argsort) = probably_coalesced_indices_hash.sort();
      return std::make_tuple(sorted, argsort.to(nnz_arange.scalar_type()));
    }
  }();

  // Perform hash intersection
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

    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .add_owned_output(dummy.expand_as(source_arange))
      .add_input(source_arange)
      .build();

    AT_DISPATCH_INDEX_TYPES(source_arange.scalar_type(), NAME, [&]() {
        const auto* RESTRICT ptr_indices = source_indices.data_ptr<index_t>();
        const auto* RESTRICT ptr_sorted_hash = sorted_hash.data_ptr<hash_t>();
        const auto sorted_hash_len = sorted_hash.numel();
        const auto* RESTRICT ptr_hash_coeffs = hash_coeffs.template data_ptr<hash_t>();
        auto* RESTRICT ptr_intersection_count = intersection_count.data_ptr<hash_t>();
        auto* RESTRICT ptr_intersection_first_idx = intersection_first_idx.data_ptr<hash_t>();
        // non-const because of gcc-5/clang-5 issues
        auto sdim = static_cast<uint32_t>(sparse_dim);

        // Fusing hash computation with hash intersection.
        KernelLauncher::launch(iter,
            // Windows does not seem to like these nested captures without explicit names.
            [ptr_indices, ptr_sorted_hash, sorted_hash_len, ptr_hash_coeffs,
              ptr_intersection_count, ptr_intersection_first_idx, sdim,
              indices_dim_stride, indices_nnz_stride]
            FUNCAPI (index_t nnz_idx) -> index_t {
            // Compute hash value
            const auto* RESTRICT ptr_indices_dim = ptr_indices + nnz_idx * indices_nnz_stride;
            auto hash = hash_t {0};
            for (uint32_t dim = 0; dim < sdim; ++dim) {
              // Use only int32_t operations when hash_t == int32_t.
              const auto dim_hash_coeff = ptr_hash_coeffs[dim];
              const auto dim_index = static_cast<hash_t>(ptr_indices_dim[dim * indices_dim_stride]);
              hash += dim_index * dim_hash_coeff;
            }

            // Perform hash values intersection
            const auto* RESTRICT lb = find_bound<const hash_t*, hash_t, /*is_lower=*/true>(
                ptr_sorted_hash,
                ptr_sorted_hash + sorted_hash_len,
                hash
            );

            const auto* RESTRICT ub = find_bound<const hash_t*, hash_t, /*is_lower=*/false>(
                ptr_sorted_hash,
                ptr_sorted_hash + sorted_hash_len,
                hash
            );

            ptr_intersection_count[nnz_idx] = ub - lb;
            ptr_intersection_first_idx[nnz_idx] = lb - ptr_sorted_hash;

            return 0;
        });
    });

    return std::make_tuple(intersection_count, intersection_first_idx);
  }();

  Tensor selected_source, selected_probably_coalesced;
  std::tie(selected_source, selected_probably_coalesced) = [&]() -> std::tuple<Tensor, Tensor> {
    // Thread offset = shifted_offset - shift.
    // This computation is fused in kernels below.

    // hash_t might not be enough to store offset values, so we use
    // offset_t which is at least sizeof(hash_t).
    const auto kOffset = std::is_same<offset_t, int32_t>::value ? kInt : kLong;
    const auto shifted_offset = intersection_count.cumsum(-1, kOffset);

    // NOTE: unavoidable sync to get to know the result's shape.
    const auto intersection_nnz = static_cast<int64_t>(
        // shifted_offset is a 1-dim tensor, potentially empty
        shifted_offset.size(0)
        ? shifted_offset.select(-1, -1).template item<offset_t>()
        : 0);

    auto selected_buffer = at::empty({2, intersection_nnz}, intersection_count.options());
    auto selected_source = selected_buffer.select(0, 0);
    auto selected_probably_coalesced = selected_buffer.select(0, 1);
    const auto source_idx = nnz_arange.narrow(-1, 0, source._nnz());
    auto dummy = at::empty({1}, source_idx.options());

    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .add_owned_output(dummy.expand_as(source_idx))
      .add_input(source_idx) // index_t
      .add_input(intersection_count) // hash_t
      .add_input(intersection_first_idx) // hash_t
      .add_input(shifted_offset) // offset_t
      .build();

    AT_DISPATCH_INDEX_TYPES(source_idx.scalar_type(), NAME, [&]() {
        auto* RESTRICT ptr_selected_source = selected_source.data_ptr<hash_t>();
        auto* RESTRICT ptr_selected_probably_coalesced = selected_probably_coalesced.data_ptr<hash_t>();
        const auto* RESTRICT ptr_argsort = argsort_hash.data_ptr<index_t>();
        KernelLauncher::launch(iter,
            // Windows does not seem to like these nested captures without explicit names.
            [ptr_selected_source, ptr_selected_probably_coalesced, ptr_argsort]
            FUNCAPI (
              index_t idx,
              hash_t count,
              hash_t first_match_idx,
              offset_t shifted_offset) -> index_t {
            const auto offset = shifted_offset - static_cast<offset_t>(count);
            auto* RESTRICT ptr_selected_source_idx_out = ptr_selected_source + offset;
            auto* RESTRICT ptr_selected_probably_coalesced_idx_out = ptr_selected_probably_coalesced + offset;
            const auto* RESTRICT ptr_argsort_idx = ptr_argsort + first_match_idx;
            for (hash_t i = 0; i < count; ++i) {
              *ptr_selected_source_idx_out++ = idx;
              *ptr_selected_probably_coalesced_idx_out++ = *ptr_argsort_idx++;
            }

            return 0;
        });
    });

    return std::make_tuple(selected_source, selected_probably_coalesced);
  }();

  const auto res_indices = source._indices().index_select(1, selected_source);
  const auto selected_source_values = source._values().index_select(0, selected_source);
  const auto selected_probably_coalesced_values = probably_coalesced._values().index_select(0, selected_probably_coalesced);
  const auto res_values = binary_op_t::apply(selected_source_values, selected_probably_coalesced_values)
    // no-op for out-of-place calls, but we still need to cast when the op is supposed to be performed in-place
    // but binary_op_t promotes types. For example, let the op == mul, x.dtype == int8, y.dtype == uint8,
    // then mul(x, y).dtype == int16, while x.mul_(y).dtype == int8 and y.mul_(x).dtype == uint8.
    .to(res.scalar_type());
  const auto res_sparse_dim = source.sparse_dim();
  const auto res_dense_dim = res_values.dim() - 1;
  const auto res_shape = broadcasted_shape;
  const auto res_nnz = selected_source_values.size(0);

  get_sparse_impl(res)->raw_resize_(res_sparse_dim, res_dense_dim, res_shape);
  get_sparse_impl(res)->set_indices_and_values_unsafe(res_indices, res_values);
  get_sparse_impl(res)->set_nnz_and_narrow(res_nnz);
  // Result is coalesced iff arguments are coalesced, conditioned on the fact
  // that we do not check that intersection hash values are sorted and unique.
  // <= : intersection contains only unique indices (or empty), and the algorithm's
  // behavior is order-preserving. So, the result has only unique indices (or empty) which are sorted.
  // => : proof by contraposition. The contrapositive statement reads
  // `there is an uncoalesced argument => result is not coalesced`.
  // If both arguments are uncoalesced, the result is clearly uncoalesced again
  // thanks to the order-preserving behavior of the algorithm.
  // Otherwise we have a coalesced argument `probably_coalesced` and an uncoalesced `source`.
  // Since the matching beahavior of the algorithm respects the order of `source`, the result
  // will be as coalesced as `source` is, which is uncoalesced.
  res._coalesced_(source.is_coalesced() && probably_coalesced.is_coalesced());

  return res;
}

template <
  template <typename func_t> class kernel_t,
  typename binary_op_t>
Tensor& _sparse_binary_op_intersection_kernel_out(
    Tensor& res,
    const Tensor& x,
    const Tensor& y,
    const bool is_commutative = true
) {
  TORCH_CHECK(
      (x.is_sparse() && y.is_sparse())
      && (x.dim() == y.dim()) && (x.sparse_dim() == y.sparse_dim())
      && (x.sizes().slice(0, x.sparse_dim()) == y.sizes().slice(0, y.sparse_dim())),
      NAME, "(): expects sparse inputs with equal dimensionality, ",
      "number of sparse dimensions, and shape of sparse dimensions");

  const auto broadcasted_shape = infer_size(x.sizes(), y.sizes());

  int64_t max_hash_val = 1;
  for (const auto d : c10::irange(x.sparse_dim())) {
    max_hash_val *= broadcasted_shape[d];
  }

  // Optimization: use 32-bit hash values when possible.
  const auto is_max_hash_32bits = max_hash_val <= std::numeric_limits<int>::max();
  // Intersection nnz could get larger than nnz of either arguments.
  // Example: probably_coalesced and source have only one unique and shared index,
  // then the size of intersection is exactly the product of their nnzs.
  // This nnz defines offsets per thread which are computed using cumsum on values
  // of hash dtype. This becomes a problem when hash_t=int32_t and res_nnz > max(int32_t).
  const auto is_max_offset_32bits = (x._nnz() * y._nnz()) <= std::numeric_limits<int>::max();

  if (is_max_hash_32bits && is_max_offset_32bits) {
    using hash_t = int32_t;
    using offset_t = int32_t;
    return _sparse_binary_op_intersection_kernel_impl<kernel_t, binary_op_t, hash_t, offset_t>(
        res, x, y, broadcasted_shape, is_commutative);
  }
  else if (is_max_hash_32bits && !is_max_offset_32bits) {
    using hash_t = int32_t;
    using offset_t = int64_t;
    return _sparse_binary_op_intersection_kernel_impl<kernel_t, binary_op_t, hash_t, offset_t>(
        res, x, y, broadcasted_shape, is_commutative);
  }
  else if (!is_max_hash_32bits && is_max_offset_32bits) {
    using hash_t = int64_t;
    using offset_t = int32_t;
    return _sparse_binary_op_intersection_kernel_impl<kernel_t, binary_op_t, hash_t, offset_t>(
        res, x, y, broadcasted_shape, is_commutative);
  }
  else {
    using hash_t = int64_t;
    using offset_t = int64_t;
    return _sparse_binary_op_intersection_kernel_impl<kernel_t, binary_op_t, hash_t, offset_t>(
        res, x, y, broadcasted_shape, is_commutative);
  }
}

} // anonymous namespace

}} // at::native
