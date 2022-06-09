#include <ATen/Dispatch.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/util/ArrayRef.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/sparse_coo_tensor.h>
#endif

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/logical.h>

namespace at {
namespace native {
using namespace at::sparse;
using at::cuda::detail::getTensorInfo;
using at::cuda::detail::TensorInfo;

/******************************************************************************
 * Build sparse from diagonals
 ******************************************************************************/

// --------------------------------------------------------------------
// spdiags(D, O, (N,M)) -> S
//
// Take rows of D and place them on the diagonals specified by offsets O of a
// new (NxM) sparse matrix S If D is (P x Q) then O must be a row vector (P, ).
// It does not matter if Q values fit  on any diagonal of S, or if S has no
// O[i]th diagonal (those values/diagonals are simply skipped)
// --------------------------------------------------------------------

namespace {

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void _spdiags_sparse_coo_cuda_kernel(
    int64_t total_diags,
    const TensorInfo<int64_t, int64_t> offsets_ti,
    const TensorInfo<scalar_t, int64_t> diagonals_ti,
    const TensorInfo<int64_t, int64_t> nnz_per_diag_ti,
    const TensorInfo<int64_t, int64_t> nnz_prefix_ti,
    TensorInfo<int64_t, int64_t> indices_out_ti,
    TensorInfo<scalar_t, int64_t> values_out_ti) {
  // First input diagonal to handle
  const int64_t d_0 = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t diag_stride0 = diagonals_ti.strides[0];
  int64_t indices_stride0 = indices_out_ti.strides[0];
  int64_t grid_stride = blockDim.x * gridDim.x;

  for (int64_t di = d_0; di < total_diags; di += grid_stride) {
    int64_t n_out = nnz_per_diag_ti.data[di];
    if (n_out > 0) {
      int64_t offset = offsets_ti.data[di];
      int64_t col_begin = ::max(offset, int64_t{0});
      int64_t row_begin = col_begin - offset;
      int64_t read_begin = (diag_stride0 * di) + col_begin;
      int64_t write_begin = nnz_prefix_ti.data[di];
      for (int64_t i = 0; i < n_out; ++i) {
        values_out_ti.data[write_begin + i] = diagonals_ti.data[read_begin + i];
        indices_out_ti.data[write_begin + i] = row_begin + i;
        indices_out_ti.data[indices_stride0 + write_begin + i] = col_begin + i;
      }
    }
  }
}
} // namespace

SparseTensor spdiags_sparse_cuda(
    const Tensor& diagonals,
    const Tensor& offsets,
    IntArrayRef shape,
    c10::optional<Layout> layout) {
  TORCH_CHECK(diagonals.dim() == 2, "Diagonals must be 2d");
  TORCH_CHECK(shape.size() == 2, "Output shape must be 2d");
  TORCH_CHECK(offsets.dim() == 1, "Offsets must be 1d");
  TORCH_CHECK(
      diagonals.size(0) == offsets.size(0),
      "Number of diagonals (",
      diagonals.size(0),
      ") does not match the number of offsets (",
      offsets.size(0),
      ")");
  if (layout) {
    TORCH_CHECK(
        (*layout == Layout::Sparse) || (*layout == Layout::SparseCsc) ||
            (*layout == Layout::SparseCsr),
        "Only output layouts (",
        Layout::Sparse,
        ", ",
        Layout::SparseCsc,
        ", and ",
        Layout::SparseCsr,
        ") are supported");
  }

  const int64_t n_row_out = shape[0];
  const int64_t n_col_out = shape[1];
  const int64_t n_col_in = diagonals.size(1);
  const int64_t n_diag = diagonals.size(0);

  auto offsets_cont = offsets.contiguous();
  auto diagonals_cont = diagonals.contiguous();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::cuda::ThrustAllocator alloc;
  auto policy = thrust::cuda::par(alloc).on(stream);
  {
    auto off_begin = offsets_cont.data_ptr<int64_t>();
    auto off_end = off_begin + offsets_cont.size(0);
    TORCH_CHECK(
        thrust::all_of(
            policy,
            off_begin,
            off_end,
            [off_begin, off_end, policy] __device__(
                const int64_t& off_value) -> bool {
              return thrust::count(policy, off_begin, off_end, off_value) == 1;
            }),
        "Duplicate offset values are not allowed");
    auto bad_off_iter = thrust::find_if_not(
        policy,
        off_begin,
        off_end,
        [n_row_out, n_col_out] __device__(const int64_t& off_value) -> bool {
          return ((-n_row_out) < off_value) && (off_value < n_col_out);
        });
    TORCH_CHECK(
        bad_off_iter == off_end,
        "Diagonal ",
        *bad_off_iter,
        " does not exist in output shape (",
        n_row_out,
        ",",
        n_col_out,
        ")");
  }

  // Contains the nnz added to output per diagonal input
  auto nnz_per_diag = at::empty_like(offsets_cont);

  thrust::transform(
      policy,
      offsets_cont.data_ptr<int64_t>(),
      offsets_cont.data_ptr<int64_t>() + n_diag,
      nnz_per_diag.data_ptr<int64_t>(),
      [n_row_out, n_col_out, n_col_in] __device__(
          const int64_t& offset) -> int64_t {
        if (offset >= 0) {
          auto col_out_lim = n_col_out - offset;
          auto col_in_lim = n_col_in - offset; // positive offsets -> read start
                                               // of in row r is shifted forward
          return ::max(
              ::min(::min(n_row_out, col_out_lim), col_in_lim), int64_t{0});
        } else {
          auto row_lim = n_row_out + offset;
          return ::max(::min(::min(row_lim, n_col_out), n_col_in), int64_t{0});
        }
      });
  // Contains the starting position along the nnz length axis of values/indices
  // we will write into
  auto nnz_prefix_sum = at::empty_like(offsets_cont);
  thrust::exclusive_scan(
      policy,
      nnz_per_diag.data_ptr<int64_t>(),
      nnz_per_diag.data_ptr<int64_t>() + n_diag,
      nnz_prefix_sum.data_ptr<int64_t>(),
      0);

  int64_t nnz = thrust::reduce(
      policy,
      nnz_per_diag.data_ptr<int64_t>(),
      nnz_per_diag.data_ptr<int64_t>() + n_diag);

  auto indices_out = at::empty({2, nnz}, offsets.options());
  auto values_out = at::empty({nnz}, diagonals.options());
  auto max_length_diag = std::min(std::min(n_row_out, n_col_out), n_col_in);
  if (nnz > 0) {
    auto indices_out_ti = getTensorInfo<int64_t, int64_t>(indices_out);
    auto offsets_ti = getTensorInfo<int64_t, int64_t>(offsets_cont);
    auto nnz_per_diag_ti = getTensorInfo<int64_t, int64_t>(nnz_per_diag);
    auto nnz_prefix_sum_ti = getTensorInfo<int64_t, int64_t>(nnz_prefix_sum);
    int64_t block_size = std::min(
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
    auto grid_size = ceil_div(nnz, block_size);
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::ComplexHalf,
        values_out.scalar_type(),
        "spdiags_sparse_cuda",
        [&] {
          auto values_out_ti = getTensorInfo<scalar_t, int64_t>(values_out);
          auto diagonals_ti = getTensorInfo<scalar_t, int64_t>(diagonals_cont);
          _spdiags_sparse_coo_cuda_kernel<scalar_t><<<grid_size, block_size, 0, stream>>>(
              n_diag,
              offsets_ti,
              diagonals_ti,
              nnz_per_diag_ti,
              nnz_prefix_sum_ti,
              indices_out_ti,
              values_out_ti);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
  SparseTensor result_coo = at::sparse_coo_tensor(
      indices_out,
      values_out,
      shape,
      diagonals.options().layout(Layout::Sparse));
  if (layout) {
    if (*layout == Layout::SparseCsr) {
      return result_coo.to_sparse_csr();
    }
    if (*layout == Layout::SparseCsc) {
      return result_coo.to_sparse_csc();
    }
  }
  return result_coo;
}

} // namespace native
} // namespace at
