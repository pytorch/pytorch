#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Dispatch.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPUCC
#define FUNCAPI __host__ __device__
#define INLINE __forceinline__
#define NAME "compressed_index_invariance_checks_cuda"
#else
#define FUNCAPI
#define INLINE inline
#define NAME "compressed_index_invariance_checks_cpu"
#endif

#if defined(_WIN32) || defined(_WIN64)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

#define INVARIANT_CHECK_FUNC_API static INLINE FUNCAPI void

namespace at {
namespace native {

namespace {

// NOTE: all the checks but the very last one are designed
// to work with vectors.
// To enable vectorization one would need to write a conversion
// Vec -> bool and make kernel launchers call into vectorized
// execution paths.

// All the invariants are described in https://pearu.github.io/bsr_tensor_invariants.html

INVARIANT_CHECK_FUNC_API
_assert(const bool cond, const char* const message) {
#ifdef GPUCC
  CUDA_KERNEL_ASSERT(cond && message);
#else
  TORCH_CHECK(cond, message);
#endif
}

// Invariant 5.1
// cidx[..., 0] == 0
template <typename index_t>
INVARIANT_CHECK_FUNC_API
_check_first_cidx_is_zero(const index_t& cidx, const index_t& zero) {
  const bool invariant = cidx == zero;
  static constexpr auto message = "`c{row|col}_indices[..., 0] == 0` is not satisfied.";
  _assert(invariant, message);
}

// Invariant 5.2
// cidx[..., -1] == nnz
template <typename index_t>
INVARIANT_CHECK_FUNC_API
_check_last_cidx_is_nnz(const index_t& cidx, const index_t& nnz) {
  const bool invariant = cidx == nnz;
  static constexpr auto message = "`c{row|col}_indices[..., -1] == nnz` is not satisfied.";
  _assert(invariant, message);
}

// Invariant 5.3
// 0 <= cidx[..., 1:] - cidx[..., :-1] <= dim,
// where cidx/dim is either crow/ncols or ccol/nrows.
template <typename index_t>
INVARIANT_CHECK_FUNC_API
_check_cidx_nondecreasing_locally_bounded_sequence(
    const index_t& cidx,
    const index_t& cidx_next,
    const index_t& zero,
    const index_t& dim) {
  const auto s_cidx = cidx_next - cidx;
  const bool invariant = zero <= s_cidx && s_cidx <= dim;
  static constexpr auto message = "`0 <= c{row|col}_indices[..., 1:] - c{row|col}_indices[..., :-1] <= dim` is not satisfied.";
  _assert(invariant, message);
}

// Invariants 5.4 and 5.5
// 0 <= idx < dim,
// where idx/dim is either col/ncols or row/nrows.
template <typename index_t>
INVARIANT_CHECK_FUNC_API
_check_idx_bounds(
    const index_t& idx,
    const index_t& zero,
    const index_t& dim) {
  const bool invariant = zero <= idx && idx < dim;
  static constexpr auto message = "`0 <= {row|col}_indices < dim` is not satisfied.";
  _assert(invariant, message);
}

// Invariant 5.6
// {col/row}_indices[..., c{row|col}[..., i - 1]:c{row|col}[..., i]]
// for all i = 1, ..., cdim
// are sorted and distinct along the last dimension values.
template <typename index_t>
INVARIANT_CHECK_FUNC_API
_check_idx_sorted_distinct_vals_slices_with_cidx(
    const index_t* RESTRICT ptr_idx_batch,
    const index_t cidx,
    const index_t cidx_next) {
  static constexpr auto message = "`{col|row}_indices[..., c{row|col}_indices[..., i - 1]:c{row|col}_indices[..., i]] "
                       "for all i = 1, ..., cdim "
                       "are sorted and distinct along the last dimension values` "
                       "is not satisfied.";
  // Note that ptr_idx_batch = &idx[batch_idx] and is contiguous.
  const auto* RESTRICT slice_begin = ptr_idx_batch + cidx;
  const auto* RESTRICT slice_end = ptr_idx_batch + cidx_next;
  for (auto* RESTRICT curr = slice_begin + 1; curr < slice_end; ++curr) {
    const auto invariant = *(curr - 1) < *curr;
    _assert(invariant, message);
  }
}

static inline int64_t numel(IntArrayRef sizes) {
  int64_t res = 1;
  for (const auto& s : sizes) {
    res *= s;
  }
  return res;
}

template <typename func_t, typename vec_func_t>
struct EmptyVecKernel {
  static void launch(TensorIteratorBase& iter, const func_t& f, const vec_func_t& vec_f) {
  }
};

template <typename scalar_t>
using DummyVec = scalar_t;

template <
  template <typename func_t> class kernel_t,
  template <typename func_t, typename vec_func_t> class vec_kernel_t>
struct KernelLauncher {
  template <typename func_t, typename vec_func_t>
  static void launch(TensorIteratorBase& iter, const func_t& f, const vec_func_t& vec_f) {
    vec_kernel_t<func_t, vec_func_t>::launch(iter, f, vec_f);
  }

  template <typename func_t>
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    kernel_t<func_t>::launch(iter, f);
  }
};

template <
  template <typename func_t> class kernel_t,
  template <typename func_t, typename vec_func_t> class vec_kernel_t = EmptyVecKernel,
  template <typename scalar_t> class Vec = DummyVec>
void validate_compressed_sparse_indices_kernel(
    const Tensor& cidx,
    const Tensor& idx,
    const int64_t cdim,
    const int64_t dim,
    const int64_t nnz) {
  TORCH_CHECK(cidx.size(-1) == cdim + 1, "c{row|col}_indices have wrong shape: ",
      "c{row|col}.shape[-1] = ", cidx.size(-1), " is not equal to ",
      "cdim + 1 = ", cdim + 1);
  TORCH_CHECK(idx.size(-1) == nnz, "{row|col}_indices have wrong shape: ",
      "{row|col}.shape[-1] = ", idx.size(-1), " is not equal to ",
      "nnz = ", nnz);

  using KernelLauncher = KernelLauncher<kernel_t, vec_kernel_t>;

  // For TensorIterator's output: no void lambdas.
  const auto dummy = at::empty({1}, cidx.options());

  // Invariants 5.4 and 5.5
  {
    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .add_owned_output(dummy.expand_as(idx))
      .add_input(idx)
      .build();

    AT_DISPATCH_INDEX_TYPES(idx.scalar_type(), NAME, [&iter, dim] () {
        const auto zero = index_t {0};
        KernelLauncher::launch(iter,
            [zero, dim] FUNCAPI (index_t idx) -> index_t {
              _check_idx_bounds<index_t>(idx, zero, dim);
              return 0;
            }
        );
    });
  }

  // Invariants 5.1, 5.2, 5.3, 5.6
  {
    const auto cidx_first = cidx.slice(-1, 0, 1);
    const auto cidx_last = cidx.slice(-1, cdim, cdim + 1);

    const auto cidx_curr = cidx.slice(-1, 0, cdim);
    const auto cidx_next = cidx.slice(-1, 1, cdim + 1);

    const auto batch_dims = cidx.sizes().slice(0, cidx.dim() - 1);
    const auto batch_count = numel(batch_dims);
    const auto batch_idx = at::arange(batch_count, cidx.options()).view(batch_dims).unsqueeze_(-1);

    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .add_owned_output(dummy.expand_as(cidx_curr))
      .add_input(cidx_first)
      .add_input(cidx_last)
      .add_input(cidx_curr)
      .add_input(cidx_next)
      .add_input(batch_idx)
      .build();

    AT_DISPATCH_INDEX_TYPES(idx.scalar_type(), NAME, [&iter, &idx, dim, nnz] () {
        const auto* RESTRICT ptr_idx = idx.data_ptr<index_t>();
        const auto zero = index_t {0};
        KernelLauncher::launch(iter,
            [zero, dim, nnz, ptr_idx] FUNCAPI (
              index_t cidx_first,
              index_t cidx_last,
              index_t cidx_curr,
              index_t cidx_next,
              index_t batch_idx) -> index_t {
              // Invariant 5.1
              _check_first_cidx_is_zero<index_t>(cidx_first, zero);
              // Invariant 5.2
              _check_last_cidx_is_nnz<index_t>(cidx_last, nnz);
              // Invariant 5.3
              _check_cidx_nondecreasing_locally_bounded_sequence<index_t>(cidx_curr, cidx_next, zero, dim);
              // Invariant 5.6
              // NOTE: the implementation below is sync-less, but, unfortunately,
              // work is not guaranteed to be well-balanced between different threads.
              // idx is contiguous and of shape (..., nnz), so batches are multiples of nnz apart.
              const auto* RESTRICT ptr_idx_batch = ptr_idx + batch_idx * nnz;
              _check_idx_sorted_distinct_vals_slices_with_cidx<index_t>(ptr_idx_batch, cidx_curr, cidx_next);
              return 0;
            }
        );
    });
  }
}

} // anonymous namespace for invariance checkers and utilities

}}
