#pragma once
#include <ATen/NumericUtils.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/CUDAContext.h>

#include <c10/util/Load.h>
#include <limits>
#include <cmath>

namespace at {
namespace native {

template <typename integer>
constexpr inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

template <typename integer>
constexpr inline integer get_log_num_threads_x_inner_scan(integer num_rows, integer row_size) {
  integer log_num_threads_x = 0;
  integer log_num_threads_y = 0;
  while (((integer)1 << log_num_threads_x) < row_size) {
    ++log_num_threads_x;
  }
  while (((integer)1 << log_num_threads_y) < num_rows) {
    ++log_num_threads_y;
  }
  // we want to keep the ratio between the x-threads and y-threads about the same as
  // the ratio between the row_size and num_rows, but the total number of threads in
  // a block should be about 512
  integer diff = log_num_threads_x - log_num_threads_y;
  // 9 is from log2(512)
  log_num_threads_x = ((integer)9 + diff) / (integer)2;
  // I found that in having larger log_num_threads_x can give significant speed up in some cases,
  // but detrimental in another case, so just keep the lower bound to be log2(16) == 4 to make it
  // similar to the previous implementation
  // Keeping the upper bound to be log2(512) == 9 as the maximum number of threads in a block.
  log_num_threads_x = std::min(std::max((integer)4, log_num_threads_x), (integer)9);
  return log_num_threads_x;
}

template<typename scalar_t, typename idx_t, typename BinaryOperation>
__device__ void binary_op_update(const scalar_t lhs, scalar_t& rhs, const idx_t lhs_idx, idx_t& rhs_idx, BinaryOperation binary_op) {
  if(!at::_isnan(rhs) && (at::_isnan(lhs) || !binary_op(rhs, lhs))) {
    rhs = lhs;
    rhs_idx = lhs_idx;
  }
}
/* Perform an inclusive scan along the innermost dimension of a tensor.
 *
 * - num_rows is the size of the flattened outer dimensions;
 * - row_size is the size of the innermost dimension;
 *
 * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is
 * considered as having 'num_rows' rows of size 'row_size'.
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
template<typename scalar_t, class BinaryFunction>
__global__ void tensor_kernel_scan_innermost_dim_with_indices(const scalar_t *self_, scalar_t *values_, int64_t *indices_,
                                                int num_rows, int row_size,
                                                const uint32_t num_threads, const uint32_t log_num_threads_x,
                                                scalar_t init, BinaryFunction binary_op) {
  // dynamic memory allocation for vbuf and ibuf
  alignas(sizeof(double)) extern __shared__ char buf[];
  scalar_t* vbuf = reinterpret_cast<scalar_t*>(buf); // the size is num_threads * 2
  int64_t* ibuf = reinterpret_cast<int64_t*>(vbuf + num_threads * 2);
  const uint32_t num_threads_x = 1 << log_num_threads_x;
  scalar_t* row_buf = vbuf + 2 * num_threads_x * threadIdx.y;
  int64_t* row_idx_buf = ibuf + 2 * num_threads_x * threadIdx.y;

  for (int block_row = blockIdx.x * blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    int row = block_row + threadIdx.y;
    const scalar_t *row_self = self_ + row * row_size;
    scalar_t *row_values = values_ + row * row_size;
    int64_t *row_indices = indices_ + row * row_size;
    scalar_t block_total = init;
    int64_t block_idx_final = 0;
    const bool row_exists = row < num_rows;
    // Perform scan on one block at a time, keeping track of the total value of
    // all blocks processed so far.
    for (int block_col = 0; block_col < row_size; block_col += 2 * num_threads_x) {
      // Load data into shared memory (two values per thread).
      int col1 = block_col + threadIdx.x;
      int col2 = block_col + num_threads_x + threadIdx.x;
      if (row_exists) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = c10::load(&row_self[col1]);
          row_idx_buf[threadIdx.x] = col1;
        } else {
          row_buf[threadIdx.x] = init;
          // No need to set the index here as the value in init will never be selected
        }

        if (col2 < row_size) {
          row_buf[num_threads_x + threadIdx.x] = c10::load(&row_self[col2]);
          row_idx_buf[num_threads_x + threadIdx.x] = col2;
        } else {
          row_buf[num_threads_x + threadIdx.x] = init;
          // No need to set the index here as the value in init will never be selected
        }

        // Add the total value of all previous blocks to the first value of this block.
        if (threadIdx.x == 0) {
          binary_op_update(block_total, row_buf[0], block_idx_final, row_idx_buf[0], binary_op);
        }
      }
      __syncthreads();

      // Parallel reduction with Sklansky method. The diagram can be seen on this paper:
      // https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back
      for (uint32_t s = 1; s <= num_threads_x; s <<= 1) {
        if (row_exists) {
          uint32_t a = (threadIdx.x / s) * (2 * s) + s;
          uint32_t ti = a + (threadIdx.x % s);
          uint32_t si = a - 1;
          binary_op_update(row_buf[si], row_buf[ti], row_idx_buf[si], row_idx_buf[ti], binary_op);
        }
        __syncthreads();
      }

      // Write back to output.
      if (row_exists) {
        if (col1 < row_size){
          row_values[col1] = row_buf[threadIdx.x];
          row_indices[col1] = row_idx_buf[threadIdx.x];
        }
        if (col2 < row_size) {
          row_values[col2] = row_buf[num_threads_x + threadIdx.x];
          row_indices[col2] = row_idx_buf[num_threads_x + threadIdx.x];
        }
      }
      block_total = row_buf[2 * num_threads_x - 1];
      block_idx_final = row_idx_buf[2 * num_threads_x - 1];
      __syncthreads();
    }
  }
}

/* Perform an inclusive scan along an outer dimension of a tensor.
 *
 * - num_orows is the size of the flattened outer dimensions;
 * - num_irows is the size of the flattened inner dimensions;
 * - row_size is the size of the dimension along which to compute the variance;
 *
 * The dimensions to the outside and inside of the specified dimension are considered as flattened.
 * Thread blocks with the same blockIdx.y process an "outer row" (i.e. an element of the flattened
 * outer dimensions, which contains several "inner rows").
 * Each thread processes a single inner row at a time.
 */
template<typename scalar_t, class BinaryFunction>
__global__ void tensor_kernel_scan_outer_dim_with_indices(const scalar_t *self_, scalar_t *values_, int64_t *indices_,
                  const uint32_t num_orows, const uint32_t num_irows, const uint32_t row_size, scalar_t init, BinaryFunction binary_op) {
  for (uint32_t orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (uint32_t irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      const scalar_t *self = self_ + orow * row_size * num_irows + irow;
      scalar_t *values = values_ + orow * row_size * num_irows + irow;
      int64_t *indices = indices_ + orow * row_size * num_irows + irow;
      scalar_t out = init;
      int64_t out_idx = 0;

      for (auto col = decltype(row_size){0}; col < row_size; ++col) {
        const auto val = c10::load(self);
        if(at::_isnan(val) || (!at::_isnan(out) && binary_op(val, out))) {
          out = val;
          out_idx = col;
        }
        *values = out;
        *indices = out_idx;
        self += num_irows;
        values += num_irows;
        indices += num_irows;
      }
    }
  }
}

inline void check_fits_in_unsigned(int64_t val, const char* name) {
  constexpr auto umax = std::numeric_limits<uint32_t>::max();
  TORCH_CHECK(
      val >= 0 && val <= umax, name, " must fit in a 32-bit uint32_t value");
}


template<typename scalar_t, class BinaryFunction>
__host__ void scan_outer_dim_with_indices(
    const TensorBase& self, const TensorBase& values, const TensorBase& indices,
    int dim, scalar_t init, BinaryFunction binary_op) {
  int64_t row_size = self.size(dim);
  auto sizes = self.sizes();

  // Treat all outer dimensions (i.e. dim_ < dim) as one.
  const int64_t num_orows = c10::multiply_integers(sizes.begin(), sizes.begin() + dim);

  // Treat all inner dimensions (i.e. dim > dimension) as one.
  const int64_t num_irows = c10::multiply_integers(sizes.begin() + dim + 1, sizes.end());
  //for performance reasons, cuda kernels use uint32_t for loops over irows, orows and row,
  //make sure that input is not bigger than supported by uint32_t
  check_fits_in_unsigned(num_irows, "num_irows");
  check_fits_in_unsigned(num_orows, "num_orows");
  check_fits_in_unsigned(row_size, "row_size");


  dim3 threads(std::min(512, int(num_irows)));
  int64_t maxGridDim = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
  dim3 grid(std::min(maxGridDim, num_orows), std::min(maxGridDim, ceil_div(num_irows, int64_t{threads.x})));
  tensor_kernel_scan_outer_dim_with_indices<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    self.const_data_ptr<scalar_t>(), values.mutable_data_ptr<scalar_t>(), indices.mutable_data_ptr<int64_t>(),
    num_orows, num_irows, row_size, init, binary_op);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, class BinaryFunction>
__host__ void scan_innermost_dim_with_indices(
    const TensorBase& self, const TensorBase& values, const TensorBase& indices,
    scalar_t init, BinaryFunction binary_op) {
  int ndim = self.dim();
  // Treat all outer dimensions as a single dimension.
  int row_size = self.size(ndim - 1);
  int num_rows = self.numel() / row_size;

  // assuming max_num_threads per block is 512
  const uint32_t num_threads = 512;
  const uint32_t log_num_threads_x = get_log_num_threads_x_inner_scan<uint32_t>(num_rows, row_size);
  const uint32_t num_threads_x = (1 << log_num_threads_x);
  const uint32_t num_threads_y = num_threads / num_threads_x;
  dim3 threads(num_threads_x, num_threads_y);
  dim3 grid(std::min(at::cuda::getCurrentDeviceProperties()->maxGridSize[0], ceil_div(num_rows, int(threads.y))));

  const uint32_t mem_size = 2 * num_threads * (sizeof(scalar_t) + sizeof(int64_t));
  tensor_kernel_scan_innermost_dim_with_indices<scalar_t><<<grid, threads, mem_size,
                                                            at::cuda::getCurrentCUDAStream()>>>(
    self.const_data_ptr<scalar_t>(), values.mutable_data_ptr<scalar_t>(), indices.mutable_data_ptr<int64_t>(),
    num_rows, row_size, num_threads, log_num_threads_x, init, binary_op);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename scalar_t, typename BinaryFunction>
void scan_dim_with_indices(const TensorBase& self, const TensorBase& values, const TensorBase& indices, //int64_t dim) {
     int64_t dim, scalar_t init, BinaryFunction binary_op) {
  int ndim = self.dim();
  auto self_ = self.expect_contiguous();
  TORCH_INTERNAL_ASSERT(values.is_contiguous() && indices.is_contiguous());
  if (dim == ndim - 1) {
    scan_innermost_dim_with_indices<scalar_t>(*self_, values, indices, init, binary_op);
  } else {
    scan_outer_dim_with_indices<scalar_t>(*self_, values, indices, dim, init, binary_op);
  }
}

// TODO: The implementation of `tensor_kernel_scan_outer_dim` and
// `tensor_kernel_scan_innermost_dim` is similar to
// `tensor_kernel_scan_outer_dim_with_indices`
// `tensor_kernel_scan_outer_dim_with_indices` and should be refactored to
// remove the duplication.

/* Perform an inclusive scan along an outer dimension of a tensor.
 *
 * - num_orows is the size of the flattened outer dimensions;
 * - num_irows is the size of the flattened inner dimensions;
 * - row_size is the size of the dimension along which to scan;
 *
 * The dimensions to the outside and inside of the specified dimension are considered as flattened.
 * Thread blocks with the same blockIdx.y process an "outer row" (i.e. an element of the flattened
 * outer dimensions, which contains several "inner rows").
 * Each thread processes a single inner row at a time.
 */
template<typename scalar_t, class BinaryOp>
__global__ void tensor_kernel_scan_outer_dim(scalar_t *tgt_, const scalar_t *src_,
                                              const uint32_t num_orows, const uint32_t num_irows, const uint32_t row_size,
                                              const scalar_t init, BinaryOp binary_op)
{
  for (uint32_t orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (uint32_t irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      const scalar_t *src = src_ + orow * row_size * num_irows + irow;
      scalar_t *tgt = tgt_ + orow * row_size * num_irows + irow;
      scalar_t acc = init;

      for (uint32_t col = 0; col < row_size; ++col) {
        acc = binary_op(acc, c10::load(src));
        *tgt = acc;

        src += num_irows;
        tgt += num_irows;
      }
    }
  }
}

/* Perform an inclusive scan along the innermost dimension of a tensor.
 *
 * - num_rows is the size of the flattened outer dimensions;
 * - row_size is the size of the innermost dimension;
 *
 * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is
 * considered as having 'num_rows' rows of size 'row_size'.
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
template<typename T, class BinaryFunction>
__device__ void tensor_kernel_scan_innermost_dim_impl(T* row_buf, T *tgt_, const T *src_,
                                      const uint32_t num_rows, const uint32_t row_size,
                                      const uint32_t log_num_threads_x,
                                      T init, BinaryFunction binary_op){
  const uint32_t num_threads_x = 1 << log_num_threads_x;
  for (uint32_t block_row = blockIdx.x * blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    uint32_t row = block_row + threadIdx.y;
    T block_total = init;

    const T *row_src = src_ + row * row_size;
    T *row_tgt = tgt_ + row * row_size;
    const bool row_exists = row < num_rows;

    // Perform scan on one block at a time, keeping track of the total value of
    // all blocks processed so far.
    for (uint32_t block_col = 0; block_col < row_size; block_col += 2 * num_threads_x) {
      // Load data into shared memory (two values per thread).
      uint32_t col1 = block_col + threadIdx.x;
      uint32_t col2 = block_col + num_threads_x + threadIdx.x;
      if (row_exists) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = row_src[col1];
        } else {
          row_buf[threadIdx.x] = init;
        }

        if (col2 < row_size) {
          row_buf[num_threads_x + threadIdx.x] = row_src[col2];
        } else {
          row_buf[num_threads_x + threadIdx.x] = init;
        }

        // Add the total value of all previous blocks to the first value of this block.
        if (threadIdx.x == 0) {
          row_buf[0] = binary_op(row_buf[0], block_total);
        }
      }
      __syncthreads();

      // Parallel reduction with Sklansky method. The diagram can be seen on this paper:
      // https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back
      for (uint32_t m = 0; m <= log_num_threads_x; ++m) {
        if (row_exists) {
          uint32_t s = 1 << m; // s = 2 ^ m
          uint32_t a = ((threadIdx.x >> m) << (m + 1)) | s; // a = (threadIdx.x / s) * (2 * s) + s
          uint32_t ti = a + (threadIdx.x % s);
          uint32_t si = a - 1;
          row_buf[ti] = binary_op(row_buf[ti], row_buf[si]);
        }
        __syncthreads();
      }

      // Write back to output.
      if (row_exists) {
        if (col1 < row_size) row_tgt[col1] = row_buf[threadIdx.x];
        if (col2 < row_size) row_tgt[col2] = row_buf[num_threads_x + threadIdx.x];
      }
      block_total = row_buf[2 * num_threads_x - 1];
      __syncthreads();
    }
  }
}

template <
    typename T,
    class BinaryFunction>
__global__ void tensor_kernel_scan_innermost_dim(
    T* tgt_,
    const T* src_,
    const uint32_t num_rows,
    const uint32_t row_size,
    const uint32_t log_num_threads_x,
    T init,
    BinaryFunction binary_op) {
  alignas(sizeof(double)) extern __shared__ char sbuf[];
  T* sbuf2 = reinterpret_cast<T*>(sbuf);
  const uint32_t num_threads_x = 1 << log_num_threads_x;
  T* row_buf = reinterpret_cast<T*>(sbuf2 + num_threads_x * 2 * threadIdx.y);

  tensor_kernel_scan_innermost_dim_impl<T>(
      row_buf, tgt_, src_, num_rows, row_size, log_num_threads_x, init, binary_op);
}


template<typename scalar_t, class BinaryFunction>
__host__ void scan_outer_dim(const TensorBase& self, const TensorBase& result,
                             int dim, scalar_t init, BinaryFunction binary_op) {
  const int64_t row_size = self.size(dim);
  auto sizes = self.sizes();

  // Treat all outer dimensions (i.e. dim_ < dim) as one.
  const int64_t num_orows = c10::multiply_integers(sizes.begin(), sizes.begin() + dim);

  // Treat all inner dimensions (i.e. dim > dimension) as one.
  const int64_t num_irows = c10::multiply_integers(sizes.begin() + dim + 1, sizes.end());

  dim3 threads(std::min(512, int(num_irows)));
  int64_t maxGridDim = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
  dim3 grid(std::min(maxGridDim, num_orows), std::min(maxGridDim, ceil_div(num_irows, int64_t{threads.x})));

  check_fits_in_unsigned(num_irows, "num_irows");
  check_fits_in_unsigned(num_orows, "num_orows");
  check_fits_in_unsigned(row_size, "row_size");

  tensor_kernel_scan_outer_dim<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    result.mutable_data_ptr<scalar_t>(), self.const_data_ptr<scalar_t>(),
    num_orows, num_irows, row_size, init, binary_op);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, class BinaryFunction>
void scan_innermost_dim(const TensorBase& self, const TensorBase& result,
                        scalar_t init, BinaryFunction binary_op) {
  int64_t ndim = self.dim();
  // Treat all outer dimensions as a single dimension.
  int64_t row_size = self.size(ndim - 1);
  int64_t num_rows = self.numel() / row_size;

  // assuming max_num_threads per block is 512
  const uint32_t num_threads = 512;
  const uint32_t log_num_threads_x = get_log_num_threads_x_inner_scan<uint32_t>(num_rows, row_size);
  const uint32_t num_threads_x = (1 << log_num_threads_x);
  const uint32_t num_threads_y = num_threads / num_threads_x;
  dim3 threads(num_threads_x, num_threads_y);
  int64_t maxGridDim = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  dim3 grid(std::min(maxGridDim, ceil_div(num_rows, int64_t{threads.y})));

  check_fits_in_unsigned(num_rows, "Number of rows (self.numel()/self.size(self.dim()-1))");
  check_fits_in_unsigned(row_size, "row_size");

  tensor_kernel_scan_innermost_dim<scalar_t><<<grid, threads, num_threads * 2 * sizeof(scalar_t),
                                               at::cuda::getCurrentCUDAStream()>>>(
    result.mutable_data_ptr<scalar_t>(), self.const_data_ptr<scalar_t>(),
    num_rows, row_size, log_num_threads_x, init, binary_op);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename scalar_t, typename BinaryFunction>
void scan_dim(const TensorBase& self, const TensorBase& result,
     int64_t dim, scalar_t init, BinaryFunction binary_op) {
  int ndim = self.dim();
  auto self_ = self.expect_contiguous();
  TORCH_INTERNAL_ASSERT(result.is_contiguous());

  if (self.numel() == self.size(dim)) {
    cuda::cub::inclusive_scan(self_->const_data_ptr<scalar_t>(), result.mutable_data_ptr<scalar_t>(), binary_op, self.numel());
  } else if (dim == ndim - 1) {
    scan_innermost_dim<scalar_t>(*self_, result, init, binary_op);
  } else {
    scan_outer_dim<scalar_t>(*self_, result, dim, init, binary_op);
  }
}

}}  // namespace at::native
