#pragma once
#include <ATen/NumericUtils.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/macros/Macros.h>
#include <c10/util/Load.h>
#include <limits>
#include <cmath>
#include <cstring>

namespace at::native {

template <typename integer>
constexpr inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

constexpr inline uint32_t get_log_num_threads_x_inner_scan(int64_t num_rows, int64_t row_size) {
  int64_t log_num_threads_x = 0;
  int64_t log_num_threads_y = 0;
  while ((int64_t{1} << log_num_threads_x) < row_size) {
    ++log_num_threads_x;
  }
  while ((int64_t{1} << log_num_threads_y) < num_rows) {
    ++log_num_threads_y;
  }
  // we want to keep the ratio between the x-threads and y-threads about the same as
  // the ratio between the row_size and num_rows, but the total number of threads in
  // a block should be about 512
  const int64_t diff = log_num_threads_x - log_num_threads_y;
  // 9 is from log2(512)
  log_num_threads_x = (int64_t{9} + diff) / int64_t{2};
  // I found that in having larger log_num_threads_x can give significant speed up in some cases,
  // but detrimental in another case, so just keep the lower bound to be log2(16) == 4 to make it
  // similar to the previous implementation
  // Keeping the upper bound to be log2(512) == 9 as the maximum number of threads in a block.
  log_num_threads_x = std::clamp(log_num_threads_x, int64_t{4}, int64_t{9});
  return static_cast<uint32_t>(log_num_threads_x);
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
  const uint32_t log_num_threads_x = get_log_num_threads_x_inner_scan(num_rows, row_size);
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
template<typename scalar_t, typename index_t, class BinaryOp>
__global__ void tensor_kernel_scan_outer_dim(scalar_t *tgt_, const scalar_t *src_,
                                              const uint32_t num_orows, const uint32_t num_irows, const uint32_t row_size,
                                              const scalar_t init, BinaryOp binary_op)
{
  for (uint32_t orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (uint32_t irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      const scalar_t *src = src_ + static_cast<index_t>(orow) * row_size * num_irows + irow;
      scalar_t *tgt = tgt_ + (index_t) orow * row_size * num_irows + irow;
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
template<typename T, typename index_t, class BinaryFunction>
__device__ void tensor_kernel_scan_innermost_dim_impl(T* row_buf, T *tgt_, const T *src_,
                                      const uint32_t num_rows, const uint32_t row_size,
                                      const uint32_t log_num_threads_x,
                                      T init, BinaryFunction binary_op){
  const index_t num_threads_x = 1 << log_num_threads_x;
  for (index_t block_row = blockIdx.x * (index_t) blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    index_t row = block_row + (index_t) threadIdx.y;
    T block_total = init;

    const T *row_src = src_ + row * row_size;
    T *row_tgt = tgt_ + row * row_size;
    const bool row_exists = row < num_rows;

    // Perform scan on one block at a time, keeping track of the total value of
    // all blocks processed so far.
    for (index_t block_col = 0; block_col < row_size; block_col += 2 * num_threads_x) {
      // Load data into shared memory (two values per thread).
      index_t col1 = block_col + (index_t) threadIdx.x;
      index_t col2 = block_col + num_threads_x + (index_t) threadIdx.x;
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
      for (int m = 0; m <= log_num_threads_x; ++m) {
        if (row_exists) {
          index_t s = 1 << m; // s = 2 ^ m
          auto a = static_cast<index_t>((threadIdx.x >> m) << (m + 1)) | s; // a = (threadIdx.x / s) * (2 * s) + s
          index_t ti = a + (threadIdx.x % s);
          index_t si = a - 1;
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
  if (num_rows * (size_t) row_size <= UINT_MAX) {
      tensor_kernel_scan_innermost_dim_impl<T, uint32_t>(
          row_buf, tgt_, src_, num_rows, row_size, log_num_threads_x, init, binary_op);
  } else {
      tensor_kernel_scan_innermost_dim_impl<T, size_t>(
          row_buf, tgt_, src_, num_rows, row_size, log_num_threads_x, init, binary_op);
  }
}

/* Row-parallel inclusive scan along the innermost dimension.
 *
 * Each thread loads kItems contiguous elements (vectorized when aligned), scans
 * them in registers, and combines per-thread totals with warp shuffles, so a
 * tile needs at most one __syncthreads() instead of log2(num_threads_x) + 1.
 * Depending on the shape, a row is handled by a group of <= 32 lanes
 * (scan_rows_warp), by a whole block (scan_rows_block), or split into chunks
 * across several blocks (reduce_row_chunks + scan_rows_block) so that a few
 * long rows can still fill the GPU.
 */
namespace inner_scan {

constexpr int kBlock = 256;
// Logical warp width used for shuffles; valid on both wave32 and wave64 hardware.
constexpr int kLogicalWarp = 32;
// Bytes each thread keeps in flight in reduce_row_chunks.
constexpr int kReduceBytesInFlight = 128;

template <typename T, typename BinaryFunction>
struct Traits {
  // cumsum and cumprod (std::plus / std::multiplies) are memory-bound, so a thread takes 32 bytes.
  // Other ops (logcumsumexp) are compute-bound on 8- and 16-byte types, where a thread applies
  // binary_op about (2 * kItems + 6) / kItems times per element, so those take 16 items, as does
  // every 16-byte type (on 32-lane waves; see launch).
  static constexpr bool kCheapOp =
      std::is_same_v<BinaryFunction, std::plus<T>> || std::is_same_v<BinaryFunction, std::multiplies<T>>;
  static constexpr int kItems =
      (sizeof(T) >= 16 || (sizeof(T) >= 8 && !kCheapOp)) ? 16 : static_cast<int>(32 / sizeof(T));
  static constexpr int kVec = std::min<int>(kItems, sizeof(T) >= 16 ? 1 : 16 / sizeof(T));
  // Half / bfloat16 sums and products round after every combine, so their error grows with the
  // serial chain inside a thread; scanning the items as runs of 8 halves that chain.
  static constexpr bool kHalf = std::is_same_v<T, c10::Half> || std::is_same_v<T, c10::BFloat16>;
  static constexpr int kRun = (kHalf && kCheapOp) ? 8 : kItems;
  static_assert(kItems % kVec == 0 && kItems % kRun == 0);
};

template <typename T, int N, size_t A = sizeof(T) * N>
struct alignas(A) Vec {
  T v[N];
};

// Global memory is accessed through GVec, which only assumes 4-byte (or element) alignment:
// multi-dword loads and stores need just dword alignment, so an input that starts a few elements
// into its storage is still read with 16-byte accesses.
template <typename T>
constexpr size_t kGlobalAlign = alignof(T) > 4 ? alignof(T) : 4;
template <typename T, int N>
using GVec = Vec<T, N, kGlobalAlign<T>>;

template <typename T, typename F>
__device__ __forceinline__ T shfl_words(const T& v, F f) {
  constexpr int W = (sizeof(T) + sizeof(int) - 1) / sizeof(int);
  int w[W] = {};
  memcpy(w, &v, sizeof(T));
#pragma unroll
  for (int i = 0; i < W; ++i) {
    w[i] = f(w[i]);
  }
  T r;
  memcpy(&r, w, sizeof(T));
  return r;
}

template <typename T>
__device__ __forceinline__ T shfl_up(const T& v, unsigned delta, int width) {
  return shfl_words(v, [=](int x) { return WARP_SHFL_UP(x, delta, width); });
}

template <typename T>
__device__ __forceinline__ T shfl_down(const T& v, unsigned delta, int width) {
  return shfl_words(v, [=](int x) { return WARP_SHFL_DOWN(x, delta, width); });
}

template <typename T>
__device__ __forceinline__ T shfl_idx(const T& v, int src_lane, int width) {
  return shfl_words(v, [=](int x) { return WARP_SHFL(x, src_lane, width); });
}

// width must be a power of two <= kLogicalWarp.
template <typename T, typename BinaryFunction>
__device__ __forceinline__ T warp_inclusive_scan(T v, int lane, int width, BinaryFunction binary_op) {
  for (int offset = 1; offset < width; offset <<= 1) {
    const T other = shfl_up(v, offset, width);
    if (lane >= offset) {
      v = binary_op(other, v);
    }
  }
  return v;
}

template <int ITEMS, int VEC, typename T>
__device__ __forceinline__ void load_items(T (&x)[ITEMS], const T* p, uint64_t col, uint64_t end,
                                           bool valid, bool vec_ok, T fill) {
  if (valid && vec_ok && col + ITEMS <= end) {
#pragma unroll
    for (int i = 0; i < ITEMS / VEC; ++i) {
      const auto t = *reinterpret_cast<const GVec<T, VEC>*>(p + col + i * VEC);
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        x[i * VEC + j] = t.v[j];
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
      x[j] = (valid && col + j < end) ? p[col + j] : fill;
    }
  }
}

template <int ITEMS, int VEC, typename T>
__device__ __forceinline__ void store_items(const T (&x)[ITEMS], T* p, uint64_t col, uint64_t end,
                                            bool valid, bool vec_ok) {
  if (!valid) {
    return;
  }
  if (vec_ok && col + ITEMS <= end) {
#pragma unroll
    for (int i = 0; i < ITEMS / VEC; ++i) {
      GVec<T, VEC> t;
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        t.v[j] = x[i * VEC + j];
      }
      *reinterpret_cast<GVec<T, VEC>*>(p + col + i * VEC) = t;
    }
  } else {
#pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
      if (col + j < end) {
        p[col + j] = x[j];
      }
    }
  }
}

// Inclusive scan of x in registers: runs of RUN items are scanned independently, then each run
// is offset by the inclusive total of the runs before it. RUN == ITEMS is the plain serial scan.
template <int RUN, int ITEMS, typename T, class BinaryFunction>
__device__ __forceinline__ void thread_scan(T (&x)[ITEMS], BinaryFunction binary_op) {
#pragma unroll
  for (int r = 0; r < ITEMS; r += RUN) {
#pragma unroll
    for (int j = 1; j < RUN; ++j) {
      x[r + j] = binary_op(x[r + j - 1], x[r + j]);
    }
  }
#pragma unroll
  for (int r = RUN; r < ITEMS; r += RUN) {
    const T before = x[r - 1];
#pragma unroll
    for (int j = 0; j < RUN; ++j) {
      x[r + j] = binary_op(before, x[r + j]);
    }
  }
}

// One row per group of blockDim.x (<= kLogicalWarp) lanes; no shared memory or barriers.
template <typename T, int ITEMS, int VEC, int RUN, class BinaryFunction>
C10_LAUNCH_BOUNDS_1(kBlock)
__global__ void scan_rows_warp(T* __restrict__ tgt, const T* __restrict__ src,
                               const uint32_t num_rows, const uint32_t row_size,
                               T init, BinaryFunction binary_op, const bool vec_ok) {
  const int group = blockDim.x;
  const int lane = threadIdx.x;
  const uint64_t tile = static_cast<uint64_t>(group) * ITEMS;
  for (uint64_t block_row = static_cast<uint64_t>(blockIdx.x) * blockDim.y; block_row < num_rows;
       block_row += static_cast<uint64_t>(blockDim.y) * gridDim.x) {
    const uint64_t row = block_row + threadIdx.y;
    const bool valid = row < num_rows;
    const T* row_src = src + row * row_size;
    T* row_tgt = tgt + row * row_size;
    T carry = init;
    for (uint64_t base = 0; base < row_size; base += tile) {
      const uint64_t col = base + static_cast<uint64_t>(lane) * ITEMS;
      T x[ITEMS];
      load_items<ITEMS, VEC>(x, row_src, col, row_size, valid, vec_ok, init);
      thread_scan<RUN>(x, binary_op);
      const T incl = warp_inclusive_scan(x[ITEMS - 1], lane, group, binary_op);
      const T excl = shfl_up(incl, 1, group);
      const T prefix = lane == 0 ? carry : binary_op(carry, excl);
#pragma unroll
      for (int j = 0; j < ITEMS; ++j) {
        x[j] = binary_op(prefix, x[j]);
      }
      store_items<ITEMS, VEC>(x, row_tgt, col, row_size, valid, vec_ok);
      carry = binary_op(carry, shfl_idx(incl, group - 1, group));
    }
  }
}

// One (row, chunk) per block. With partials != nullptr, chunk c starts from the
// fold of partials[row][0 .. c-1] written by reduce_row_chunks.
template <typename T, int ITEMS, int VEC, int RUN, class BinaryFunction>
C10_LAUNCH_BOUNDS_1(kBlock)
__global__ void scan_rows_block(T* __restrict__ tgt, const T* __restrict__ src,
                                const uint32_t num_rows, const uint32_t row_size,
                                const uint32_t chunk_size, const uint32_t num_chunks,
                                const T* __restrict__ partials,
                                T init, BinaryFunction binary_op, const bool vec_ok) {
  constexpr int kWarps = kBlock / kLogicalWarp;
  alignas(T) __shared__ char smem[2 * kWarps * sizeof(T)];
  T* warp_totals = reinterpret_cast<T*>(smem);
  const int lane = threadIdx.x % kLogicalWarp;
  const int warp = threadIdx.x / kLogicalWarp;
  const uint64_t tile = static_cast<uint64_t>(kBlock) * ITEMS;
  const uint64_t total_work = static_cast<uint64_t>(num_rows) * num_chunks;
  int buf = 0;
  for (uint64_t work = blockIdx.x; work < total_work; work += gridDim.x) {
    const uint64_t row = work / num_chunks;
    const uint32_t chunk = static_cast<uint32_t>(work % num_chunks);
    const uint64_t begin = static_cast<uint64_t>(chunk) * chunk_size;
    const uint64_t end = begin + chunk_size < row_size ? begin + chunk_size : row_size;
    const T* row_src = src + row * row_size;
    T* row_tgt = tgt + row * row_size;
    T carry = init;
    if (partials != nullptr && chunk > 0) {
      // Fold the earlier chunks' totals across the whole block (binary_op is commutative
      // whenever partials are used); takes a turn of the warp_totals double buffer.
      const T* row_partials = partials + row * (num_chunks - 1);
      T acc = init;
      for (uint32_t c = threadIdx.x; c < chunk; c += kBlock) {
        acc = binary_op(acc, row_partials[c]);
      }
#pragma unroll
      for (int offset = kLogicalWarp / 2; offset > 0; offset >>= 1) {
        acc = binary_op(acc, shfl_down(acc, offset, kLogicalWarp));
      }
      T* totals = warp_totals + buf * kWarps;
      if (lane == 0) {
        totals[warp] = acc;
      }
      __syncthreads();
      carry = totals[0];
#pragma unroll
      for (int i = 1; i < kWarps; ++i) {
        carry = binary_op(carry, totals[i]);
      }
      buf ^= 1;
    }
    for (uint64_t base = begin; base < end; base += tile) {
      const uint64_t col = base + static_cast<uint64_t>(threadIdx.x) * ITEMS;
      T x[ITEMS];
      load_items<ITEMS, VEC>(x, row_src, col, end, true, vec_ok, init);
      thread_scan<RUN>(x, binary_op);
      const T incl = warp_inclusive_scan(x[ITEMS - 1], lane, kLogicalWarp, binary_op);
      // Double-buffered so that one barrier per tile is enough.
      T* totals = warp_totals + buf * kWarps;
      if (lane == kLogicalWarp - 1) {
        totals[warp] = incl;
      }
      __syncthreads();
      T running = carry;
      T warp_prefix = carry;
#pragma unroll
      for (int i = 0; i < kWarps; ++i) {
        if (i == warp) {
          warp_prefix = running;
        }
        running = binary_op(running, totals[i]);
      }
      const T excl = shfl_up(incl, 1, kLogicalWarp);
      const T prefix = lane == 0 ? warp_prefix : binary_op(warp_prefix, excl);
#pragma unroll
      for (int j = 0; j < ITEMS; ++j) {
        x[j] = binary_op(prefix, x[j]);
      }
      store_items<ITEMS, VEC>(x, row_tgt, col, end, true, vec_ok);
      carry = running;
      buf ^= 1;
    }
  }
}

// Per-thread fold of [begin, end), where end - begin is a whole number of tiles.
// Loads are striped across the block, and kReduceUnroll tiles are issued before
// any is consumed. Tiles past `end` re-load the first tile of the group (always
// in bounds) and are discarded, so there is no data-dependent branch.
template <bool kVectorized, typename T, int ITEMS, int VEC, class BinaryFunction>
__device__ __forceinline__ T reduce_tiles(const T* __restrict__ p, uint64_t begin, uint64_t end,
                                          T init, BinaryFunction binary_op) {
  constexpr int kTileBytes = static_cast<int>(ITEMS * sizeof(T));
  constexpr int kReduceUnroll = kTileBytes >= kReduceBytesInFlight ? 1 : kReduceBytesInFlight / kTileBytes;
  constexpr int kWidth = kVectorized ? VEC : 1;
  constexpr int kLoads = ITEMS / kWidth;
  constexpr uint64_t tile = static_cast<uint64_t>(kBlock) * ITEMS;
  T acc[kReduceUnroll];
#pragma unroll
  for (int u = 0; u < kReduceUnroll; ++u) {
    acc[u] = init;
  }
  for (uint64_t base = begin; base < end; base += kReduceUnroll * tile) {
    GVec<T, kWidth> v[kReduceUnroll][kLoads];
#pragma unroll
    for (int u = 0; u < kReduceUnroll; ++u) {
      const uint64_t tile_base = base + u * tile < end ? base + u * tile : base;
#pragma unroll
      for (int l = 0; l < kLoads; ++l) {
        v[u][l] = *reinterpret_cast<const GVec<T, kWidth>*>(
            p + tile_base + static_cast<uint64_t>(l * kBlock + threadIdx.x) * kWidth);
      }
    }
#pragma unroll
    for (int u = 0; u < kReduceUnroll; ++u) {
      const bool live = base + u * tile < end;
#pragma unroll
      for (int l = 0; l < kLoads; ++l) {
#pragma unroll
        for (int k = 0; k < kWidth; ++k) {
          const T r = binary_op(acc[u], v[u][l].v[k]);
          acc[u] = live ? r : acc[u];
        }
      }
    }
  }
  T total = acc[0];
#pragma unroll
  for (int u = 1; u < kReduceUnroll; ++u) {
    total = binary_op(total, acc[u]);
  }
  return total;
}

// Totals of every chunk except the last one of each row. Requires a commutative
// binary_op and chunk_size to be a multiple of kBlock * ITEMS.
template <typename T, int ITEMS, int VEC, class BinaryFunction>
C10_LAUNCH_BOUNDS_1(kBlock)
__global__ void reduce_row_chunks(T* __restrict__ partials, const T* __restrict__ src,
                                  const uint32_t num_rows, const uint32_t row_size,
                                  const uint32_t chunk_size, const uint32_t num_chunks,
                                  T init, BinaryFunction binary_op, const bool vec_ok) {
  constexpr int kWarps = kBlock / kLogicalWarp;
  alignas(T) __shared__ char smem[kWarps * sizeof(T)];
  T* warp_sums = reinterpret_cast<T*>(smem);
  const int lane = threadIdx.x % kLogicalWarp;
  const int warp = threadIdx.x / kLogicalWarp;
  const uint32_t chunks_to_reduce = num_chunks - 1;
  const uint64_t total_work = static_cast<uint64_t>(num_rows) * chunks_to_reduce;
  for (uint64_t work = blockIdx.x; work < total_work; work += gridDim.x) {
    // Walk chunks backwards so the scan pass, which walks forwards, first re-reads
    // the chunks that are most likely still in L2 / Infinity Cache.
    const uint64_t item = total_work - 1 - work;
    const uint64_t row = item / chunks_to_reduce;
    const uint32_t chunk = static_cast<uint32_t>(item % chunks_to_reduce);
    const uint64_t begin = static_cast<uint64_t>(chunk) * chunk_size;
    const uint64_t end = begin + chunk_size;
    const T* row_src = src + row * row_size;
    T acc = vec_ok
        ? reduce_tiles<true, T, ITEMS, VEC>(row_src, begin, end, init, binary_op)
        : reduce_tiles<false, T, ITEMS, VEC>(row_src, begin, end, init, binary_op);
#pragma unroll
    for (int offset = kLogicalWarp / 2; offset > 0; offset >>= 1) {
      acc = binary_op(acc, shfl_down(acc, offset, kLogicalWarp));
    }
    if (lane == 0) {
      warp_sums[warp] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      T total = warp_sums[0];
#pragma unroll
      for (int i = 1; i < kWarps; ++i) {
        total = binary_op(total, warp_sums[i]);
      }
      partials[row * chunks_to_reduce + chunk] = total;
    }
    __syncthreads();
  }
}

enum class Strategy { Warp, Block, Chunked };

struct Plan {
  Strategy strategy;
  int group = kLogicalWarp;
  int64_t num_chunks = 1;
  int64_t chunk_size = 0;
};

// Use the barrier-free warp kernel for long rows once there are this many rows per SM.
constexpr int64_t kWarpRowsPerSM = 64;
// Rows shorter than a block tile leave most of a block idle, so they get a warp each
// once there are this many rows per SM.
constexpr int64_t kShortRowWarpRowsPerSM = 16;
// Split rows across several blocks when there are fewer than this many rows per SM.
constexpr int64_t kBlocksPerSM = 4;
constexpr int64_t kMinTilesPerChunk = 2;
// Chunking adds a launch and re-reads all but the last chunk of each row; with two chunks that
// costs more than the halved serial chain saves, so such rows get one block each.
constexpr int64_t kMinChunks = 3;
constexpr int64_t kMaxChunks = 1024;

inline Plan make_plan(int64_t num_rows, int64_t row_size, int64_t num_sms, int64_t items) {
  if (row_size <= kLogicalWarp * items) {
    int group = 1;
    while (group * items < row_size) {
      group <<= 1;
    }
    return {Strategy::Warp, group};
  }
  if (num_rows >= num_sms * kWarpRowsPerSM ||
      (row_size < kBlock * items && num_rows >= num_sms * kShortRowWarpRowsPerSM)) {
    return {Strategy::Warp, kLogicalWarp};
  }
  if (num_rows < num_sms * kBlocksPerSM) {
    const int64_t block_tile = kBlock * items;
    int64_t num_chunks = std::max<int64_t>(2, ceil_div(num_sms * kBlocksPerSM, num_rows));
    num_chunks = std::min(num_chunks, row_size / (kMinTilesPerChunk * block_tile));
    num_chunks = std::min(num_chunks, kMaxChunks);
    if (num_chunks >= kMinChunks) {
      const int64_t chunk_size = ceil_div(ceil_div(row_size, num_chunks), block_tile) * block_tile;
      num_chunks = ceil_div(row_size, chunk_size);
      if (num_chunks >= kMinChunks) {
        return {Strategy::Chunked, 0, num_chunks, chunk_size};
      }
    }
  }
  return {Strategy::Block, 0, 1, row_size};
}

template <int kItems, int kVec, int kRun, typename T, class BinaryFunction>
void launch_with(T* tgt, const T* src, int64_t num_rows, int64_t row_size, T init, BinaryFunction binary_op) {
  constexpr uintptr_t kAlign = kGlobalAlign<T>;
  const bool vec_ok = row_size % kVec == 0 &&
      reinterpret_cast<uintptr_t>(src) % kAlign == 0 &&
      reinterpret_cast<uintptr_t>(tgt) % kAlign == 0;
  const int64_t num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t max_blocks = num_sms * 32;
  const Plan plan = make_plan(num_rows, row_size, num_sms, kItems);
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto rows = static_cast<uint32_t>(num_rows);
  const auto cols = static_cast<uint32_t>(row_size);
  const auto grid_for = [&](int64_t work) {
    return dim3(static_cast<uint32_t>(std::min(max_blocks, work)));
  };

  switch (plan.strategy) {
    case Strategy::Warp: {
      const dim3 block(plan.group, kBlock / plan.group);
      scan_rows_warp<T, kItems, kVec, kRun><<<grid_for(ceil_div(num_rows, int64_t{block.y})), block, 0, stream>>>(
          tgt, src, rows, cols, init, binary_op, vec_ok);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    case Strategy::Block: {
      scan_rows_block<T, kItems, kVec, kRun><<<grid_for(num_rows), kBlock, 0, stream>>>(
          tgt, src, rows, cols, cols, 1, nullptr, init, binary_op, vec_ok);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    case Strategy::Chunked: {
      const auto num_chunks = static_cast<uint32_t>(plan.num_chunks);
      const auto chunk_size = static_cast<uint32_t>(plan.chunk_size);
      auto& allocator = *c10::cuda::CUDACachingAllocator::get();
      auto partials = allocator.allocate(num_rows * (plan.num_chunks - 1) * sizeof(T));
      T* partials_ptr = static_cast<T*>(partials.get());
      reduce_row_chunks<T, kItems, kVec><<<grid_for(num_rows * (plan.num_chunks - 1)), kBlock, 0, stream>>>(
          partials_ptr, src, rows, cols, chunk_size, num_chunks, init, binary_op, vec_ok);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      scan_rows_block<T, kItems, kVec, kRun><<<grid_for(num_rows * plan.num_chunks), kBlock, 0, stream>>>(
          tgt, src, rows, cols, chunk_size, num_chunks, partials_ptr, init, binary_op, vec_ok);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
  }
}

template <typename T, class BinaryFunction>
void launch(T* tgt, const T* src, int64_t num_rows, int64_t row_size, T init, BinaryFunction binary_op) {
  using Tr = Traits<T, BinaryFunction>;
  if constexpr (sizeof(T) >= 16) {
    // On 64-lane waves, 16 contiguous 16-byte items per lane spread every load instruction over
    // 64 cache lines; there cumsum / cumprod are fastest with 2 items and other ops with 8.
    if (at::cuda::warp_size() == 64) {
      constexpr int kItems64 = Tr::kCheapOp ? 2 : 8;
      return launch_with<kItems64, 1, kItems64>(tgt, src, num_rows, row_size, init, binary_op);
    }
  }
  launch_with<Tr::kItems, Tr::kVec, Tr::kRun>(tgt, src, num_rows, row_size, init, binary_op);
}

} // namespace inner_scan

#if defined(USE_ROCM)
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
#define AT_OUTER_SCAN_TDM 1
#if defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_tensor_load_to_lds_d2) && __has_builtin(__builtin_amdgcn_tensor_store_from_lds_d2)
#define AT_OUTER_SCAN_TDM_BUILTINS 1
#endif
#if __has_builtin(__builtin_amdgcn_s_wait_tensorcnt)
#define AT_OUTER_SCAN_TDM_WAIT_BUILTIN 1
#endif
#endif
#endif

/* Tiled inclusive scan along an outer dimension.
 *
 * The tensor is viewed as [num_orows, row_size, num_irows] and scanned along row_size.
 * tensor_kernel_scan_outer_dim gives each (orow, irow) column one thread that walks the whole
 * column serially, which leaves the GPU almost idle when num_orows * num_irows is small. Here a
 * block scans (tile_h x tile_w) tiles staged in LDS with kBlock / tile_w threads per column, and
 * long columns are split into chunks: reduce_chunks writes each chunk's column totals,
 * scan_partials turns them into running totals, and scan_chunks scans every chunk starting from
 * the running total of the chunks before it.
 *
 * On gfx1250 one wave moves tiles between global memory and LDS with the Tensor Data Mover
 * (TENSOR_LOAD_TO_LDS / TENSOR_STORE_FROM_LDS), pipelined through a kStages-deep LDS ring; other
 * targets copy tiles with ordinary loads and stores.
 */
namespace outer_scan {

constexpr int kBlock = 256;
constexpr int kWarp = inner_scan::kLogicalWarp;
constexpr int kWarps = kBlock / kWarp;
constexpr int kStages = 4;
// Most chunk totals one lane of scan_partials folds; bounds the chunk count to kWarp times this.
constexpr int kPartialsPerLane = 32;
constexpr int64_t kMaxChunks = int64_t{kWarp} * kPartialsPerLane;
// Below this many rows the one-thread-per-column kernel is as fast.
constexpr int64_t kMinRows = 16;
// Split columns into chunks only if there are fewer (orow, strip) groups than SMs * this...
constexpr int64_t kChunkBlocksPerSM = 4;
// ...and each group has at least this many tiles; shorter columns don't repay the extra launches.
constexpr int64_t kMinTilesToChunk = 16;
constexpr int64_t kMinTilesPerChunk = 2;
// Two chunks cost two extra launches and a re-read of half of every column, more than halving the
// serial chain saves, so such columns stay in one piece.
constexpr int64_t kMinChunks = 3;
// Without the Tensor Data Mover tiles are copied synchronously and a block's LDS ring leaves room
// for only a block or two per CU. Once a row of the tensor holds this many bytes per SM (128 f32
// columns), the one-thread-per-column kernel already keeps the memory system busy and is faster.
constexpr int64_t kStockRowBytesPerSM = 512;

template <typename T>
struct Traits {
  static_assert(sizeof(T) <= 8, "the Tensor Data Mover moves elements of 1, 2, 4 or 8 bytes");
  // Rows of one column that a thread scans per tile. Odd, so that the threads sharing the columns
  // of a narrow tile (tile_w < kWarp) read from different LDS banks.
  static constexpr int kRows = 64 / sizeof(T) - 1;
  static constexpr int kTileElems = kBlock * kRows;
  static constexpr int kVec = 16 / sizeof(T);
  // kStages tiles, then one T per thread for combining column segments.
  static constexpr size_t kSmemBytes = (static_cast<size_t>(kStages) * kTileElems + kBlock) * sizeof(T);
};

struct Problem {
  uint32_t num_orows, row_size, num_irows;
  uint32_t log_tile_w;  // tile_w = 1 << log_tile_w <= kBlock
  uint32_t tile_h;      // (kBlock >> log_tile_w) * kRows
  uint32_t strips;      // ceil(num_irows / tile_w) column strips per orow
  uint32_t num_chunks;  // pieces each (orow, strip) column group is split into along row_size
  uint32_t chunk_rows;  // rows per piece; a multiple of tile_h
  bool vec_ok;          // 16-byte accesses are allowed (ordinary-load mover only)
};

// A tile's first element, and how much of the tensor (within the current chunk) lies below and to
// the right of it. The rest of the tile reads as zero and is never written back.
struct Tile {
  uint64_t offset;
  uint32_t rows;
  uint32_t cols;
};

template <typename T>
__device__ __forceinline__ void copy_in(T* tile, const T* __restrict__ src, const Tile& t, const Problem& p) {
  constexpr int E = Traits<T>::kTileElems;
  constexpr int V = Traits<T>::kVec;
  const uint32_t mask = (1u << p.log_tile_w) - 1;
  if (p.vec_ok && mask + 1 >= static_cast<uint32_t>(V)) {
    for (uint32_t v = threadIdx.x; v < static_cast<uint32_t>(E / V); v += kBlock) {
      const uint32_t e = v * V, r = e >> p.log_tile_w, c = e & mask;
      inner_scan::Vec<T, V> x;
      if (r < t.rows && c < t.cols) {
        const auto g = *reinterpret_cast<const inner_scan::GVec<T, V>*>(src + t.offset + static_cast<uint64_t>(r) * p.num_irows + c);
#pragma unroll
        for (int k = 0; k < V; ++k) {
          x.v[k] = g.v[k];
        }
      } else {
#pragma unroll
        for (int k = 0; k < V; ++k) {
          x.v[k] = T(0);
        }
      }
      *reinterpret_cast<inner_scan::Vec<T, V>*>(tile + e) = x;
    }
  } else {
    for (uint32_t e = threadIdx.x; e < static_cast<uint32_t>(E); e += kBlock) {
      const uint32_t r = e >> p.log_tile_w, c = e & mask;
      tile[e] = (r < t.rows && c < t.cols) ? src[t.offset + static_cast<uint64_t>(r) * p.num_irows + c] : T(0);
    }
  }
}

template <typename T>
__device__ __forceinline__ void copy_out(T* __restrict__ dst, const T* tile, const Tile& t, const Problem& p) {
  constexpr int E = Traits<T>::kTileElems;
  constexpr int V = Traits<T>::kVec;
  const uint32_t mask = (1u << p.log_tile_w) - 1;
  if (p.vec_ok && mask + 1 >= static_cast<uint32_t>(V)) {
    for (uint32_t v = threadIdx.x; v < static_cast<uint32_t>(E / V); v += kBlock) {
      const uint32_t e = v * V, r = e >> p.log_tile_w, c = e & mask;
      if (r < t.rows && c < t.cols) {
        const auto x = *reinterpret_cast<const inner_scan::Vec<T, V>*>(tile + e);
        inner_scan::GVec<T, V> g;
#pragma unroll
        for (int k = 0; k < V; ++k) {
          g.v[k] = x.v[k];
        }
        *reinterpret_cast<inner_scan::GVec<T, V>*>(dst + t.offset + static_cast<uint64_t>(r) * p.num_irows + c) = g;
      }
    }
  } else {
    for (uint32_t e = threadIdx.x; e < static_cast<uint32_t>(E); e += kBlock) {
      const uint32_t r = e >> p.log_tile_w, c = e & mask;
      if (r < t.rows && c < t.cols) {
        dst[t.offset + static_cast<uint64_t>(r) * p.num_irows + c] = tile[e];
      }
    }
  }
}

#if defined(AT_OUTER_SCAN_TDM)
typedef int tdm_i32x4 __attribute__((ext_vector_type(4)));
typedef int tdm_i32x8 __attribute__((ext_vector_type(8)));

// Groups 0 and 1 of the Tensor DMA descriptor (D#) for a 2-D tile, per "Tensor DMA Descriptor"
// in the CDNA5 ISA guide. The hardware reads the descriptor from SGPRs.
template <typename T>
__device__ __forceinline__ void tdm_descriptor(tdm_i32x4& g0, tdm_i32x8& g1, const T* gptr, const T* lds,
                                               const Tile& t, const Problem& p) {
  constexpr uint32_t log_size = sizeof(T) == 1 ? 0 : sizeof(T) == 2 ? 1 : sizeof(T) == 4 ? 2 : 3;
  const uint32_t tile_w = 1u << p.log_tile_w;
  const uint32_t rows = t.rows < p.tile_h ? t.rows : p.tile_h;
  uint32_t dim0, dim1, tile_dim0, tile_dim1;
  uint64_t stride;
  if (p.num_irows == tile_w) {
    // Rows are back to back both in memory and in LDS: move the tile as one contiguous run.
    dim0 = rows * tile_w;
    dim1 = 1;
    tile_dim0 = p.tile_h * tile_w;
    tile_dim1 = 1;
    stride = tile_dim0;
  } else {
    dim0 = t.cols;
    dim1 = rows;
    tile_dim0 = tile_w;
    tile_dim1 = p.tile_h;
    stride = p.num_irows;
  }
  const uint64_t gaddr = reinterpret_cast<uint64_t>(gptr + t.offset);
  // A flat LDS address carries the LDS byte offset in its low 32 bits.
  const uint32_t laddr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
  g0[0] = 1;  // one valid tensor
  g0[1] = static_cast<int>(laddr);
  g0[2] = static_cast<int>(static_cast<uint32_t>(gaddr));
  g0[3] = static_cast<int>((static_cast<uint32_t>(gaddr >> 32) & 0x1ffffffu) | (2u << 30));  // type = 2
  g1[0] = static_cast<int>(log_size << 16);  // no multicast, atomic barrier, iteration or padding
  g1[1] = static_cast<int>(dim0 << 16);
  g1[2] = static_cast<int>((dim0 >> 16) | (dim1 << 16));
  g1[3] = static_cast<int>((dim1 >> 16) | (tile_dim0 << 16));
  g1[4] = static_cast<int>(tile_dim1);
  g1[5] = static_cast<int>(static_cast<uint32_t>(stride));
  g1[6] = static_cast<int>(static_cast<uint32_t>(stride >> 32) & 0xffffu);
  g1[7] = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    g0[k] = __builtin_amdgcn_readfirstlane(g0[k]);
  }
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    g1[k] = __builtin_amdgcn_readfirstlane(g1[k]);
  }
}

__device__ __forceinline__ void tdm_load(tdm_i32x4 g0, tdm_i32x8 g1) {
#if defined(AT_OUTER_SCAN_TDM_BUILTINS)
  __builtin_amdgcn_tensor_load_to_lds_d2(g0, g1, 0);
#else
  asm volatile("tensor_load_to_lds %0, %1" ::"s"(g0), "s"(g1) : "memory");
#endif
}

__device__ __forceinline__ void tdm_store(tdm_i32x4 g0, tdm_i32x8 g1) {
#if defined(AT_OUTER_SCAN_TDM_BUILTINS)
  __builtin_amdgcn_tensor_store_from_lds_d2(g0, g1, 0);
#else
  asm volatile("tensor_store_from_lds %0, %1" ::"s"(g0), "s"(g1) : "memory");
#endif
}

#if defined(AT_OUTER_SCAN_TDM_WAIT_BUILTIN)
#define AT_OUTER_SCAN_WAIT_TENSORCNT(n) __builtin_amdgcn_s_wait_tensorcnt(n)
#else
#define AT_OUTER_SCAN_WAIT_TENSORCNT(n) asm volatile("s_wait_tensorcnt " #n ::: "memory")
#endif
#endif // AT_OUTER_SCAN_TDM

template <typename T>
__device__ __forceinline__ void load_tile(T* lds, const T* __restrict__ src, const Tile& t, const Problem& p) {
#if defined(AT_OUTER_SCAN_TDM)
  tdm_i32x4 g0;
  tdm_i32x8 g1;
  tdm_descriptor(g0, g1, src, lds, t, p);
  tdm_load(g0, g1);
#else
  copy_in(lds, src, t, p);
#endif
}

template <typename T>
__device__ __forceinline__ void store_tile(T* __restrict__ dst, const T* lds, const Tile& t, const Problem& p) {
#if defined(AT_OUTER_SCAN_TDM)
  tdm_i32x4 g0;
  tdm_i32x8 g1;
  tdm_descriptor(g0, g1, dst, lds, t, p);
  tdm_store(g0, g1);
#else
  copy_out(dst, lds, t, p);
#endif
}

// Waits until at most `pending` tensor ops issued by this wave are outstanding. They complete in
// issue order, so this also covers every op issued before them. Ordinary copies are synchronous.
__device__ __forceinline__ void wait_tiles(uint32_t pending) {
#if defined(AT_OUTER_SCAN_TDM)
  if (pending == 0) {
    AT_OUTER_SCAN_WAIT_TENSORCNT(0);
  } else if (pending == 1) {
    AT_OUTER_SCAN_WAIT_TENSORCNT(1);
  } else {
    AT_OUTER_SCAN_WAIT_TENSORCNT(2);
  }
#else
  (void)pending;
#endif
}

// Threads that move tiles: wave 0 with the Tensor Data Mover, everyone otherwise. Tensor
// instructions ignore EXEC, so the TDM branch must be uniform (a scalar branch).
__device__ __forceinline__ bool is_issuer() {
#if defined(AT_OUTER_SCAN_TDM)
  return __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x)) < kWarp;
#else
  return true;
#endif
}

// Thread t owns column (t % tile_w) and, within it, the kRows-row segment (t / tile_w).
struct Layout {
  uint32_t tile_w, col, seg, segs, lane;
  uint32_t lane_stride;  // lanes between consecutive segments of a column within a warp
  uint32_t q, nq;        // index / count of the warp-sized segment groups of a column
};

__device__ __forceinline__ Layout make_layout(const Problem& p) {
  Layout l;
  l.tile_w = 1u << p.log_tile_w;
  l.col = threadIdx.x & (l.tile_w - 1);
  l.seg = threadIdx.x >> p.log_tile_w;
  l.segs = kBlock >> p.log_tile_w;
  l.lane = threadIdx.x % kWarp;
  l.lane_stride = l.tile_w < static_cast<uint32_t>(kWarp) ? l.tile_w : kWarp;
  const uint32_t warps_per_seg = l.tile_w > static_cast<uint32_t>(kWarp) ? l.tile_w / kWarp : 1;
  l.q = (threadIdx.x / kWarp) / warps_per_seg;
  l.nq = kWarps / warps_per_seg;
  return l;
}

struct Work {
  uint64_t orow;
  uint32_t chunk, col0, rows, cols, num_tiles;
  uint64_t base;
};

__device__ __forceinline__ Work decode(uint64_t item, uint32_t chunks_per_group, const Problem& p) {
  Work w;
  w.chunk = static_cast<uint32_t>(item % chunks_per_group);
  const uint64_t group = item / chunks_per_group;
  const uint32_t strip = static_cast<uint32_t>(group % p.strips);
  w.orow = group / p.strips;
  w.col0 = strip << p.log_tile_w;
  const uint32_t row0 = w.chunk * p.chunk_rows;
  const uint32_t left = p.row_size - row0;
  w.rows = p.chunk_rows < left ? p.chunk_rows : left;
  w.cols = p.num_irows - w.col0;
  w.num_tiles = ceil_div(w.rows, p.tile_h);
  w.base = w.orow * p.row_size * static_cast<uint64_t>(p.num_irows) + static_cast<uint64_t>(row0) * p.num_irows + w.col0;
  return w;
}

__device__ __forceinline__ Tile tile_at(const Work& w, uint32_t t, const Problem& p) {
  return Tile{w.base + static_cast<uint64_t>(t) * p.tile_h * p.num_irows, w.rows - t * p.tile_h, w.cols};
}

// Column-wise inclusive scan of one LDS tile, continuing from `carry`: x receives this thread's
// kRows results and the column total through the end of the tile is returned. Contains a barrier;
// callers place another barrier between calls so that `tot` is free again.
template <typename T, class BinaryFunction>
__device__ __forceinline__ T scan_tile(T (&x)[Traits<T>::kRows], const T* tile, T* tot, const Layout& l, T carry,
                                       BinaryFunction binary_op) {
  constexpr int R = Traits<T>::kRows;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    x[r] = tile[(l.seg * R + r) * l.tile_w + l.col];
  }
#pragma unroll
  for (int r = 1; r < R; ++r) {
    x[r] = binary_op(x[r - 1], x[r]);
  }
  // Combine the segments of each column: first across the lanes of a warp...
  T incl = x[R - 1];
  for (uint32_t d = l.lane_stride; d < static_cast<uint32_t>(kWarp); d <<= 1) {
    const T other = inner_scan::shfl_up(incl, d, kWarp);
    if (l.lane >= d) {
      incl = binary_op(other, incl);
    }
  }
  const T warp_excl = inner_scan::shfl_up(incl, l.lane_stride, kWarp);
  // ...then across warps through LDS.
  if (l.lane + l.lane_stride >= static_cast<uint32_t>(kWarp)) {
    tot[l.q * l.tile_w + l.col] = incl;
  }
  __syncthreads();
  T running = carry;
  T prefix = carry;
#pragma unroll
  for (uint32_t i = 0; i < static_cast<uint32_t>(kWarps); ++i) {
    if (i < l.nq) {
      if (i == l.q) {
        prefix = running;
      }
      running = binary_op(running, tot[i * l.tile_w + l.col]);
    }
  }
  if (l.lane >= l.lane_stride) {
    prefix = binary_op(prefix, warp_excl);
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    x[r] = binary_op(prefix, x[r]);
  }
  return running;
}

// Column totals of every chunk except the last one of each (orow, strip), stored chunk-fastest:
// partials[orow][irow][chunk]. Requires a commutative binary_op.
template <typename T, class BinaryFunction>
C10_LAUNCH_BOUNDS_1(kBlock)
__global__ void reduce_chunks(T* __restrict__ partials, const T* __restrict__ src, const Problem p, T init,
                              BinaryFunction binary_op) {
  constexpr int R = Traits<T>::kRows;
  constexpr int E = Traits<T>::kTileElems;
  extern __shared__ __align__(16) char outer_scan_smem[];
  T* bufs = reinterpret_cast<T*>(outer_scan_smem);
  T* tot = bufs + kStages * E;
  const Layout l = make_layout(p);
  const bool issuer = is_issuer();
  const uint32_t chunks = p.num_chunks - 1;
  const uint64_t total_work = static_cast<uint64_t>(p.num_orows) * p.strips * chunks;
  for (uint64_t it = blockIdx.x; it < total_work; it += gridDim.x) {
    // Walk the work backwards so that scan_chunks, which walks forwards, first re-reads the chunks
    // most likely to still be in L2 / MALL.
    const Work w = decode(total_work - 1 - it, chunks, p);
    if (issuer) {
      for (uint32_t t = 0; t < w.num_tiles && t < static_cast<uint32_t>(kStages) - 1; ++t) {
        load_tile(bufs + t * E, src, tile_at(w, t, p), p);
      }
    }
    T acc = init;
    for (uint32_t t = 0; t < w.num_tiles; ++t) {
      if (issuer) {
        wait_tiles((t + 1 < w.num_tiles) + (t + 2 < w.num_tiles));
      }
      __syncthreads();
      if (issuer && t + 3 < w.num_tiles) {
        load_tile(bufs + ((t + 3) % kStages) * E, src, tile_at(w, t + 3, p), p);
      }
      const T* tile = bufs + (t % kStages) * E;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        acc = binary_op(acc, tile[(l.seg * R + r) * l.tile_w + l.col]);
      }
    }
    __syncthreads();
    tot[threadIdx.x] = acc;
    __syncthreads();
    if (l.seg == 0 && l.col < w.cols) {
      T total = tot[l.col];
      for (uint32_t k = 1; k < l.segs; ++k) {
        total = binary_op(total, tot[k * l.tile_w + l.col]);
      }
      partials[(w.orow * p.num_irows + w.col0 + l.col) * chunks + w.chunk] = total;
    }
    __syncthreads();
  }
}

// In place, partials[orow][irow][0 .. chunks-1] becomes its inclusive scan, so that chunk c of
// scan_chunks starts from partials[orow][irow][c - 1]. One warp per (orow, irow).
template <typename T, class BinaryFunction>
C10_LAUNCH_BOUNDS_1(kBlock)
__global__ void scan_partials(T* __restrict__ partials, const uint64_t num_columns, const uint32_t chunks, T init,
                              BinaryFunction binary_op) {
  const uint32_t lane = threadIdx.x % kWarp;
  const uint32_t per_lane = ceil_div<uint32_t>(chunks, kWarp);
  const uint32_t begin = lane * per_lane;
  for (uint64_t column = static_cast<uint64_t>(blockIdx.x) * kWarps + threadIdx.x / kWarp; column < num_columns;
       column += static_cast<uint64_t>(gridDim.x) * kWarps) {
    T* row = partials + column * chunks;
    T x[kPartialsPerLane];
    T sum = init;
#pragma unroll
    for (int k = 0; k < kPartialsPerLane; ++k) {
      if (static_cast<uint32_t>(k) < per_lane && begin + k < chunks) {
        x[k] = row[begin + k];
        sum = binary_op(sum, x[k]);
      }
    }
    const T incl = inner_scan::warp_inclusive_scan(sum, lane, kWarp, binary_op);
    const T excl = inner_scan::shfl_up(incl, 1, kWarp);
    T running = lane == 0 ? init : excl;
#pragma unroll
    for (int k = 0; k < kPartialsPerLane; ++k) {
      if (static_cast<uint32_t>(k) < per_lane && begin + k < chunks) {
        running = binary_op(running, x[k]);
        row[begin + k] = running;
      }
    }
  }
}

// Inclusive scan of every (orow, strip, chunk). With num_chunks > 1, chunk c starts from
// partials[orow][irow][c - 1] (see scan_partials).
//
// Tiles are scanned in place in a kStages-deep LDS ring. At the top of iteration t tile t has
// landed; the result of tile t-1 is then stored and tile t+2 loaded into the stage tile t-2 used.
template <typename T, class BinaryFunction>
C10_LAUNCH_BOUNDS_1(kBlock)
__global__ void scan_chunks(T* __restrict__ tgt, const T* __restrict__ src, const T* __restrict__ partials,
                            const Problem p, T init, BinaryFunction binary_op) {
  constexpr int R = Traits<T>::kRows;
  constexpr int E = Traits<T>::kTileElems;
  extern __shared__ __align__(16) char outer_scan_smem[];
  T* bufs = reinterpret_cast<T*>(outer_scan_smem);
  T* tot = bufs + kStages * E;
  const Layout l = make_layout(p);
  const bool issuer = is_issuer();
  const uint64_t total_work = static_cast<uint64_t>(p.num_orows) * p.strips * p.num_chunks;
  for (uint64_t it = blockIdx.x; it < total_work; it += gridDim.x) {
    const Work w = decode(it, p.num_chunks, p);
    if (issuer) {
      load_tile(bufs, src, tile_at(w, 0, p), p);
      if (w.num_tiles > 1) {
        load_tile(bufs + E, src, tile_at(w, 1, p), p);
      }
    }
    T carry = init;
    if (w.chunk > 0 && l.col < w.cols) {
      carry = partials[(w.orow * p.num_irows + w.col0 + l.col) * (p.num_chunks - 1) + w.chunk - 1];
    }
    for (uint32_t t = 0; t < w.num_tiles; ++t) {
      // Issued after load t: store t-2 and load t+1, both from iteration t-1.
      if (issuer) {
        wait_tiles((t >= 2) + (t + 1 < w.num_tiles));
      }
      __syncthreads();
      if (issuer) {
        if (t >= 1) {
          store_tile(tgt, bufs + ((t - 1) % kStages) * E, tile_at(w, t - 1, p), p);
        }
        if (t + 2 < w.num_tiles) {
          // That stage was last read by store t-2; issued after it: load t+1 and store t-1.
          if (t >= 2) {
            wait_tiles(2);
          }
          load_tile(bufs + ((t + 2) % kStages) * E, src, tile_at(w, t + 2, p), p);
        }
      }
      T* tile = bufs + (t % kStages) * E;
      T x[R];
      carry = scan_tile(x, tile, tot, l, carry, binary_op);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        tile[(l.seg * R + r) * l.tile_w + l.col] = x[r];
      }
    }
    __syncthreads();
    if (issuer) {
      store_tile(tgt, bufs + ((w.num_tiles - 1) % kStages) * E, tile_at(w, w.num_tiles - 1, p), p);
      wait_tiles(0);
    }
    __syncthreads();
  }
}

// gfx1250 moves tiles with the Tensor Data Mover; other ROCm targets use ordinary loads and
// stores (copy_in / copy_out).
inline bool has_tdm() {
  return std::strncmp(at::cuda::getCurrentDeviceProperties()->gcnArchName, "gfx1250", 7) == 0;
}

template <typename T>
bool should_use(int64_t num_orows, int64_t row_size, int64_t num_irows) {
  if (num_orows <= 0 || num_irows <= 0 || row_size < kMinRows) {
    return false;
  }
  if (has_tdm()) {
    return true;
  }
  const int64_t num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return num_orows * num_irows * static_cast<int64_t>(sizeof(T)) < num_sms * kStockRowBytesPerSM;
}

template <typename T>
Problem make_problem(int64_t num_orows, int64_t row_size, int64_t num_irows, int64_t num_sms, bool vec_ok) {
  Problem p{};
  p.num_orows = static_cast<uint32_t>(num_orows);
  p.row_size = static_cast<uint32_t>(row_size);
  p.num_irows = static_cast<uint32_t>(num_irows);
  uint32_t log_tile_w = 0;
  while ((int64_t{1} << log_tile_w) < num_irows && (1 << log_tile_w) < kBlock) {
    ++log_tile_w;
  }
  p.log_tile_w = log_tile_w;
  p.tile_h = static_cast<uint32_t>((kBlock >> log_tile_w) * Traits<T>::kRows);
  p.strips = static_cast<uint32_t>(ceil_div(num_irows, int64_t{1} << log_tile_w));
  const int64_t groups = num_orows * p.strips;
  const int64_t tiles = ceil_div(row_size, int64_t{p.tile_h});
  int64_t num_chunks = 1;
  if (groups < num_sms * kChunkBlocksPerSM && tiles >= kMinTilesToChunk) {
    num_chunks = std::min(std::min(ceil_div(num_sms * kChunkBlocksPerSM, groups), tiles / kMinTilesPerChunk), kMaxChunks);
  }
  if (num_chunks < kMinChunks) {
    num_chunks = 1;
  }
  const int64_t chunk_tiles = ceil_div(tiles, num_chunks);
  p.num_chunks = static_cast<uint32_t>(ceil_div(tiles, chunk_tiles));
  p.chunk_rows = static_cast<uint32_t>(chunk_tiles * p.tile_h);
  p.vec_ok = vec_ok;
  return p;
}

template <typename T, class BinaryFunction>
void launch(T* tgt, const T* src, int64_t num_orows, int64_t row_size, int64_t num_irows, T init,
            BinaryFunction binary_op) {
  constexpr size_t smem = Traits<T>::kSmemBytes;
  constexpr uintptr_t kAlign = inner_scan::kGlobalAlign<T>;
  const bool vec_ok = num_irows % Traits<T>::kVec == 0 && reinterpret_cast<uintptr_t>(src) % kAlign == 0 &&
      reinterpret_cast<uintptr_t>(tgt) % kAlign == 0;
  const int64_t num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t max_blocks = num_sms * 8;
  const Problem p = make_problem<T>(num_orows, row_size, num_irows, num_sms, vec_ok);
  const int64_t groups = num_orows * p.strips;
  auto stream = at::cuda::getCurrentCUDAStream();
  c10::DataPtr partials;
  T* partials_ptr = nullptr;
  if (p.num_chunks > 1) {
    const int64_t chunks = p.num_chunks - 1;
    partials = c10::cuda::CUDACachingAllocator::get()->allocate(num_orows * num_irows * chunks * sizeof(T));
    partials_ptr = static_cast<T*>(partials.get());
    reduce_chunks<T><<<static_cast<uint32_t>(std::min(max_blocks, groups * chunks)), kBlock, smem, stream>>>(
        partials_ptr, src, p, init, binary_op);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t columns = num_orows * num_irows;
    scan_partials<T><<<static_cast<uint32_t>(std::min(max_blocks, ceil_div(columns, int64_t{kWarps}))), kBlock, 0,
                       stream>>>(partials_ptr, static_cast<uint64_t>(columns), static_cast<uint32_t>(chunks), init,
                                 binary_op);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  scan_chunks<T><<<static_cast<uint32_t>(std::min(max_blocks, groups * p.num_chunks)), kBlock, smem, stream>>>(
      tgt, src, partials_ptr, p, init, binary_op);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace outer_scan
#endif // USE_ROCM


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
#if defined(USE_ROCM)
  if constexpr (sizeof(scalar_t) <= 8) {
    if (outer_scan::should_use<scalar_t>(num_orows, row_size, num_irows)) {
      outer_scan::launch<scalar_t>(
          result.mutable_data_ptr<scalar_t>(), self.const_data_ptr<scalar_t>(),
          num_orows, row_size, num_irows, init, binary_op);
      return;
    }
  }
#endif
  if (static_cast<size_t>(num_irows) * num_orows * row_size <= UINT_MAX) {
  tensor_kernel_scan_outer_dim<scalar_t, uint32_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    result.mutable_data_ptr<scalar_t>(), self.const_data_ptr<scalar_t>(),
    num_orows, num_irows, row_size, init, binary_op);
  } else  {
  tensor_kernel_scan_outer_dim<scalar_t, size_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    result.mutable_data_ptr<scalar_t>(), self.const_data_ptr<scalar_t>(),
    num_orows, num_irows, row_size, init, binary_op);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, class BinaryFunction>
void scan_innermost_dim(const TensorBase& self, const TensorBase& result,
                        scalar_t init, BinaryFunction binary_op) {
  int64_t ndim = self.dim();
  // Treat all outer dimensions as a single dimension.
  int64_t row_size = self.size(ndim - 1);
  int64_t num_rows = self.numel() / row_size;

#if defined(USE_ROCM)
  check_fits_in_unsigned(num_rows, "Number of rows (self.numel()/self.size(self.dim()-1))");
  check_fits_in_unsigned(row_size, "row_size");
  inner_scan::launch<scalar_t>(
      result.mutable_data_ptr<scalar_t>(), self.const_data_ptr<scalar_t>(),
      num_rows, row_size, init, binary_op);
#else
  // assuming max_num_threads per block is 512
  const uint32_t num_threads = 512;
  const uint32_t log_num_threads_x = get_log_num_threads_x_inner_scan(num_rows, row_size);
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
#endif
}

template<typename scalar_t, typename BinaryFunction>
void scan_dim(const TensorBase& self, const TensorBase& result,
     int64_t dim, scalar_t init, BinaryFunction binary_op) {
  int ndim = self.dim();
  auto self_ = self.expect_contiguous();
  TORCH_INTERNAL_ASSERT(result.is_contiguous());

  if (self.numel() == self.size(dim)) {
#if defined(USE_ROCM)
    // The chunked row scan is faster than hipCUB here and deterministic.
    if (self.numel() <= std::numeric_limits<uint32_t>::max()) {
      inner_scan::launch<scalar_t>(
          result.mutable_data_ptr<scalar_t>(), self_->const_data_ptr<scalar_t>(),
          1, self.numel(), init, binary_op);
      return;
    }
#endif
    if constexpr (std::is_same_v<BinaryFunction, std::plus<scalar_t>>) {
      if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms()) && (self.is_floating_point() || self.is_complex())) {
        cuda::cub::inclusive_deterministic_scan(self_->const_data_ptr<scalar_t>(), result.mutable_data_ptr<scalar_t>(), binary_op, self.numel());
      } else {
        cuda::cub::inclusive_scan(self_->const_data_ptr<scalar_t>(), result.mutable_data_ptr<scalar_t>(), binary_op, self.numel());
      }
    } else {
      cuda::cub::inclusive_scan(self_->const_data_ptr<scalar_t>(), result.mutable_data_ptr<scalar_t>(), binary_op, self.numel());
    }
  } else if (dim == ndim - 1) {
    scan_innermost_dim<scalar_t>(*self_, result, init, binary_op);
  } else {
    scan_outer_dim<scalar_t>(*self_, result, dim, init, binary_op);
  }
}

}  // namespace at::native
