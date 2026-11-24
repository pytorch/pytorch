#pragma once
#include <ATen/NumericUtils.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/CUDAContext.h>

#include <c10/util/Load.h>
#include <limits>
#include <cmath>

#if defined(USE_ROCM)
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/macros/Macros.h>
#include <vector>
#endif

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

#if defined(USE_ROCM)
/* Row-parallel inclusive scan along the innermost dimension.
 *
 * Each thread loads kItems contiguous elements (vectorized when aligned), scans
 * them in registers, and combines per-thread totals with warp shuffles, so a
 * tile needs at most one __syncthreads() instead of log2(num_threads_x) + 1.
 * Depending on the shape, a row is handled by a group of <= 32 lanes
 * (scan_rows_warp), by a whole block (scan_rows_block), or split into chunks
 * across several blocks (reduce_row_chunks + scan_rows_block) so that a few
 * long rows can still fill the GPU.
 * Every combine takes two adjacent ranges of a row, the earlier one first, so binary_op only
 * needs to be associative, and a partial result can overflow only where some run of the input
 * does.
 */
namespace inner_scan {

constexpr int kBlock = 256;
// Logical warp width used for shuffles; valid on both wave32 and wave64 hardware.
constexpr int kLogicalWarp = 32;
// Bytes each thread keeps in flight in reduce_row_chunks.
constexpr int kReduceBytesInFlight = 128;

template <typename T, typename BinaryFunction>
struct Traits {
  // These kernels scan cumsum and cumprod (std::plus / std::multiplies); other ops (logcumsumexp)
  // keep the stock kernels.
  static constexpr bool kCheapOp =
      std::is_same_v<BinaryFunction, std::plus<T>> || std::is_same_v<BinaryFunction, std::multiplies<T>>;
  // Both are memory-bound, so a thread takes 32 bytes; 16-byte types take 16 items (on 32-lane
  // waves; see launch).
  static constexpr int kItems = sizeof(T) >= 16 ? 16 : static_cast<int>(32 / sizeof(T));
  static constexpr int kVec = std::min<int>(kItems, sizeof(T) >= 16 ? 1 : 16 / sizeof(T));
  // Half / bfloat16 sums and products round after every combine, so their error grows with the
  // serial chain inside a thread; scanning the items as runs of 8 halves that chain.
  static constexpr bool kHalf = std::is_same_v<T, c10::Half> || std::is_same_v<T, c10::BFloat16>;
  static constexpr int kRun = kHalf ? 8 : kItems;
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

// A thread's ITEMS 1- or 2-byte elements are a whole number of 16-byte vectors, but a row of them
// can start anywhere within a dword. Unless the rows allow 16-byte accesses (vec_ok), they are
// accessed as the whole dwords that cover them, at dword-aligned addresses, and shifted into
// place, so each thread holds the same elements at any alignment and the results don't change.
// The dwords at either end are shared with the neighbouring elements (or bytes just outside the
// tensor, in the same dword as one of its elements): they are read but only written element by
// element. store_dwords needs elements aligned to their size; 2-byte elements at an odd address
// (external memory, e.g. through __cuda_array_interface__) are stored one at a time instead.
template <typename T, int ITEMS>
constexpr bool kDwordAccess = sizeof(T) < 4 && ITEMS * sizeof(T) % 16 == 0;

__device__ __forceinline__ uint32_t funnel_shift(uint32_t hi, uint32_t lo, uint32_t bytes) {
  return static_cast<uint32_t>(((static_cast<uint64_t>(hi) << 32) | lo) >> (8 * bytes));
}

// Elements are moved to and from dwords one at a time: a memcpy of the whole array makes the
// compiler place Half / BFloat16 arrays in LDS instead of registers.
template <typename T>
using ElemBits = std::conditional_t<sizeof(T) == 1, uint8_t, uint16_t>;

template <typename T>
__device__ __forceinline__ T elem_from_dwords(const uint32_t* r, int j) {
  constexpr int kPerDword = 4 / sizeof(T);
  return __builtin_bit_cast(T, static_cast<ElemBits<T>>(r[j / kPerDword] >> (8 * sizeof(T) * (j % kPerDword))));
}

template <typename T>
__device__ __forceinline__ uint32_t elem_to_dword_bits(T v, int j) {
  constexpr int kPerDword = 4 / sizeof(T);
  return static_cast<uint32_t>(__builtin_bit_cast(ElemBits<T>, v)) << (8 * sizeof(T) * (j % kPerDword));
}

// Loads the ITEMS elements at p (kFull), or the first n (0 <= n <= ITEMS) of them with the rest
// fill; p must be in bounds even for n == 0.
template <bool kFull, int ITEMS, typename T>
__device__ __forceinline__ void load_dwords(T (&x)[ITEMS], const T* p, uint32_t n, T fill) {
  constexpr int W = ITEMS * sizeof(T) / 4;
  const uint32_t m = reinterpret_cast<uintptr_t>(p) % 4;
  const uint32_t* q = reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(p) - m);
  uint32_t w[W + 1];
  if constexpr (kFull) {
#pragma unroll
    for (int i = 0; i < W / 4; ++i) {
      const auto t = *reinterpret_cast<const GVec<uint32_t, 4>*>(q + 4 * i);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        w[4 * i + k] = t.v[k];
      }
    }
    // With m == 0, w[W] only feeds a shift by 0 bytes.
    w[W] = q[m != 0 ? W : W - 1];
  } else {
    // Loads past the last dword the n elements touch re-read that dword instead, so every load is
    // in bounds without a branch around it; those words only reach elements past n.
    const uint32_t last = ((n > 0 ? n : 1) * sizeof(T) + m - 1) / 4;
#pragma unroll
    for (int i = 0; i <= W; ++i) {
      w[i] = q[i < last ? i : last];
    }
  }
  uint32_t r[W];
#pragma unroll
  for (int i = 0; i < W; ++i) {
    r[i] = funnel_shift(w[i + 1], w[i], m);
  }
#pragma unroll
  for (int j = 0; j < ITEMS; ++j) {
    x[j] = kFull || j < n ? elem_from_dwords<T>(r, j) : fill;
  }
}

template <int ITEMS, typename T>
__device__ __forceinline__ void store_dwords(const T (&x)[ITEMS], T* p) {
  constexpr int W = ITEMS * sizeof(T) / 4;
  constexpr int kPerDword = 4 / sizeof(T);
  uint32_t r[W] = {};
#pragma unroll
  for (int j = 0; j < ITEMS; ++j) {
    r[j / kPerDword] |= elem_to_dword_bits(x[j], j);
  }
  const uint32_t m = reinterpret_cast<uintptr_t>(p) % 4;
  if (m == 0) {
#pragma unroll
    for (int i = 0; i < W / 4; ++i) {
      *reinterpret_cast<GVec<uint32_t, 4>*>(reinterpret_cast<uint32_t*>(p) + 4 * i) =
          GVec<uint32_t, 4>{{r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]}};
    }
    return;
  }
  uint32_t* q = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(p) - m);
  uint32_t w[W];
#pragma unroll
  for (int i = 1; i < W; ++i) {
    w[i] = funnel_shift(r[i], r[i - 1], 4 - m);
  }
  *reinterpret_cast<GVec<uint32_t, 3>*>(q + 1) = GVec<uint32_t, 3>{{w[1], w[2], w[3]}};
#pragma unroll
  for (int i = 1; i < W / 4; ++i) {
    *reinterpret_cast<GVec<uint32_t, 4>*>(q + 4 * i) =
        GVec<uint32_t, 4>{{w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3]}};
  }
  const uint32_t lead = (4 - m) / sizeof(T);
#pragma unroll
  for (int j = 0; j < kPerDword - 1; ++j) {
    if (j < lead) {
      p[j] = x[j];
    }
    if (j < kPerDword - lead) {
      p[ITEMS - 1 - j] = x[ITEMS - 1 - j];
    }
  }
}

// Stores the first n (1 <= n < ITEMS) of the ITEMS elements to p: the dwords they cover whole
// with dword stores, the elements in the dwords at either end one at a time.
template <int ITEMS, typename T>
__device__ __forceinline__ void store_dwords_partial(const T (&x)[ITEMS], T* p, uint32_t n) {
  constexpr int W = ITEMS * sizeof(T) / 4;
  constexpr int kPerDword = 4 / sizeof(T);
  uint32_t r[W] = {};
#pragma unroll
  for (int j = 0; j < ITEMS; ++j) {
    r[j / kPerDword] |= elem_to_dword_bits(x[j], j);
  }
  const uint32_t m = reinterpret_cast<uintptr_t>(p) % 4;
  uint32_t* q = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(p) - m);
  const uint32_t bytes = m + n * sizeof(T);
  // q[first, full) lie inside the elements; q[0] (if m != 0) and q[full] (if bytes % 4 != 0)
  // are shared with the neighbouring elements.
  const uint32_t first = m != 0 ? 1 : 0;
  const uint32_t full = bytes / 4;
  uint32_t w[W + 1];
  w[0] = r[0] << (8 * m);
#pragma unroll
  for (int i = 1; i < W; ++i) {
    w[i] = funnel_shift(r[i], r[i - 1], 4 - m);
  }
  w[W] = funnel_shift(0, r[W - 1], 4 - m);
  uint32_t tail = w[0];
#pragma unroll
  for (int i = 0; i <= W; ++i) {
    if (i >= first && i < full) {
      q[i] = w[i];
    }
    if (i == full) {
      tail = w[i];
    }
  }
  const auto store_edge = [&](uint32_t d, uint32_t word) {
#pragma unroll
    for (int k = 0; k < kPerDword; ++k) {
      const uint32_t b = 4 * d + k * sizeof(T);
      if (b >= m && b < bytes) {
        p[(b - m) / sizeof(T)] = __builtin_bit_cast(T, static_cast<ElemBits<T>>(word >> (8 * sizeof(T) * k)));
      }
    }
  };
  if (m != 0) {
    store_edge(0, w[0]);
  }
  if (4 * full < bytes && (full != 0 || m == 0)) {
    store_edge(full, tail);
  }
}

// On gfx950, partial tiles of vector-aligned rows of 4-byte elements take vector loads where they can
// (see load_items): with element loads, rows of 96 to 255 elements were slower than stock. On other
// targets, and for other element sizes, the extra path makes some rows slower.
#if defined(__gfx950__)
template <typename T>
constexpr bool kVecPartialTile = sizeof(T) == 4;
#else
template <typename T>
constexpr bool kVecPartialTile = false;
#endif

// Loads the ITEMS elements at p + col, where p is the row start. kFull: the tile is whole, so every
// lane's elements are in [0, end). Otherwise elements at or past end are fill, and every lane runs
// the same branch-free loads, clamped to [0, end): lanes that take different load paths each wait
// for their own loads, which costs a memory round trip per path. The exception is kVecPartialTile,
// where only the lane that straddles end adds element loads.
template <bool kFull, int ITEMS, int VEC, typename T>
__device__ __forceinline__ void load_items(T (&x)[ITEMS], const T* p, uint64_t col, uint64_t end,
                                           bool vec_ok, T fill) {
  if (kFull && vec_ok) {
#pragma unroll
    for (int i = 0; i < ITEMS / VEC; ++i) {
      const auto t = *reinterpret_cast<const GVec<T, VEC>*>(p + col + i * VEC);
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        x[i * VEC + j] = t.v[j];
      }
    }
  } else if constexpr (kDwordAccess<T, ITEMS>) {
    if constexpr (kFull) {
      load_dwords<true>(x, p + col, ITEMS, fill);
    } else {
      const uint32_t n = col < end ? static_cast<uint32_t>(end - col < ITEMS ? end - col : ITEMS) : 0;
      load_dwords<false>(x, p + (col < end ? col : end - 1), n, fill);
    }
  } else {
    if constexpr (!kFull && kVecPartialTile<T>) {
      if (vec_ok && end >= ITEMS) {
        // Lanes whose ITEMS elements are all in [0, end) take the vector loads of a whole tile;
        // the others load the first ITEMS elements of the row (in bounds, as end >= ITEMS) in the
        // same instructions, then only the lane that straddles end loads its elements one at a time.
        const bool whole = col + ITEMS <= end;
        const uint64_t at = whole ? col : 0;
#pragma unroll
        for (int i = 0; i < ITEMS / VEC; ++i) {
          const auto t = *reinterpret_cast<const GVec<T, VEC>*>(p + at + i * VEC);
#pragma unroll
          for (int j = 0; j < VEC; ++j) {
            x[i * VEC + j] = t.v[j];
          }
        }
        if (!whole) {
#pragma unroll
          for (int j = 0; j < ITEMS; ++j) {
            x[j] = col + j < end ? p[col + j] : fill;
          }
        }
        return;
      }
    }
    // Also whole tiles without vector alignment: element loads of the same addresses as the vector
    // loads would be merged with them at element alignment.
#pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
      const T v = p[col + j < end ? col + j : end - 1];
      x[j] = col + j < end ? v : fill;
    }
  }
}

template <bool kFull, int ITEMS, int VEC, typename T>
__device__ __forceinline__ void store_items(const T (&x)[ITEMS], T* p, uint64_t col, uint64_t end,
                                            bool valid, bool vec_ok) {
  if (!valid || (!kFull && col >= end)) {
    return;
  }
  if (vec_ok && (kFull || col + ITEMS <= end)) {
#pragma unroll
    for (int i = 0; i < ITEMS / VEC; ++i) {
      GVec<T, VEC> t;
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        t.v[j] = x[i * VEC + j];
      }
      *reinterpret_cast<GVec<T, VEC>*>(p + col + i * VEC) = t;
    }
    return;
  }
  if constexpr (kDwordAccess<T, ITEMS>) {
    if (reinterpret_cast<uintptr_t>(p) % sizeof(T) == 0) {
      if (kFull || col + ITEMS <= end) {
        store_dwords(x, p + col);
      } else {
        store_dwords_partial(x, p + col, static_cast<uint32_t>(end - col));
      }
      return;
    }
  }
#pragma unroll
  for (int j = 0; j < ITEMS; ++j) {
    if (col + j < end) {
      p[col + j] = x[j];
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
__global__ void scan_rows_warp(T* tgt, const T* src,
                               const uint32_t num_rows, const uint32_t row_size,
                               T init, BinaryFunction binary_op, const bool vec_ok) {
  const int group = blockDim.x;
  const int lane = threadIdx.x;
  const uint64_t tile = static_cast<uint64_t>(group) * ITEMS;
  for (uint64_t block_row = static_cast<uint64_t>(blockIdx.x) * blockDim.y; block_row < num_rows;
       block_row += static_cast<uint64_t>(blockDim.y) * gridDim.x) {
    const uint64_t row = block_row + threadIdx.y;
    const bool valid = row < num_rows;
    // Rows past the end load the last row instead, so that no load needs a branch around it.
    const T* row_src = src + (valid ? row : num_rows - 1) * row_size;
    T* row_tgt = tgt + row * row_size;
    T carry = init;
    for (uint64_t base = 0; base < row_size; base += tile) {
      const uint64_t col = base + static_cast<uint64_t>(lane) * ITEMS;
      // The same for every lane of the block.
      const bool full = base + tile <= row_size;
      T x[ITEMS];
      if (full) {
        load_items<true, ITEMS, VEC>(x, row_src, col, row_size, vec_ok, init);
      } else {
        load_items<false, ITEMS, VEC>(x, row_src, col, row_size, vec_ok, init);
      }
      thread_scan<RUN>(x, binary_op);
      const T incl = warp_inclusive_scan(x[ITEMS - 1], lane, group, binary_op);
      const T excl = shfl_up(incl, 1, group);
      const T prefix = lane == 0 ? carry : binary_op(carry, excl);
#pragma unroll
      for (int j = 0; j < ITEMS; ++j) {
        x[j] = binary_op(prefix, x[j]);
      }
      if (full) {
        store_items<true, ITEMS, VEC>(x, row_tgt, col, row_size, valid, vec_ok);
      } else {
        store_items<false, ITEMS, VEC>(x, row_tgt, col, row_size, valid, vec_ok);
      }
      carry = binary_op(carry, shfl_idx(incl, group - 1, group));
    }
  }
}

// One (row, chunk) per block. With partials != nullptr, chunk c starts from the
// fold of partials[row][0 .. c-1] written by reduce_row_chunks.
template <typename T, int ITEMS, int VEC, int RUN, class BinaryFunction>
C10_LAUNCH_BOUNDS_1(kBlock)
__global__ void scan_rows_block(T* tgt, const T* src,
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
      // Fold the earlier chunks' totals across the whole block, each thread taking a contiguous
      // run of them and neighbours combined first; takes a turn of the warp_totals double buffer.
      const T* row_partials = partials + row * (num_chunks - 1);
      const uint32_t per_thread = ceil_div<uint32_t>(chunk, kBlock);
      const uint32_t c0 = threadIdx.x * per_thread;
      const uint32_t c1 = c0 + per_thread < chunk ? c0 + per_thread : chunk;
      T acc = init;
      for (uint32_t c = c0; c < c1; ++c) {
        acc = binary_op(acc, row_partials[c]);
      }
#pragma unroll
      for (int offset = 1; offset < kLogicalWarp; offset <<= 1) {
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
      const bool full = base + tile <= end;
      T x[ITEMS];
      if (full) {
        load_items<true, ITEMS, VEC>(x, row_src, col, end, vec_ok, init);
      } else {
        load_items<false, ITEMS, VEC>(x, row_src, col, end, vec_ok, init);
      }
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
      if (full) {
        store_items<true, ITEMS, VEC>(x, row_tgt, col, end, true, vec_ok);
      } else {
        store_items<false, ITEMS, VEC>(x, row_tgt, col, end, true, vec_ok);
      }
      carry = running;
      buf ^= 1;
    }
  }
}

// Fold of [begin, end) by one logical warp, where end - begin is a whole number of warp tiles
// (kLogicalWarp * ITEMS elements); the total ends up in lane 0. As in the scan, every partial
// is the fold of a contiguous range, so it overflows only where some run of the input does:
// each lane folds its ITEMS contiguous elements, the lanes of a warp tile are combined
// neighbours first, and warp tiles are folded in order. kReduceUnroll warp tiles are loaded
// before any is consumed; tiles past `end` re-load the first one (always in bounds) and are
// discarded, so there is no data-dependent branch.
template <bool kVectorized, typename T, int ITEMS, int VEC, class BinaryFunction>
__device__ __forceinline__ T reduce_span(const T* __restrict__ p, uint64_t begin, uint64_t end, int lane,
                                         T init, BinaryFunction binary_op) {
  constexpr int kTileBytes = static_cast<int>(ITEMS * sizeof(T));
  constexpr int kReduceUnroll = kTileBytes >= kReduceBytesInFlight ? 1 : kReduceBytesInFlight / kTileBytes;
  constexpr uint64_t tile = static_cast<uint64_t>(kLogicalWarp) * ITEMS;
  T total = init;
  for (uint64_t base = begin; base < end; base += kReduceUnroll * tile) {
    T x[kReduceUnroll][ITEMS];
#pragma unroll
    for (int u = 0; u < kReduceUnroll; ++u) {
      const uint64_t tile_base = base + u * tile < end ? base + u * tile : base;
      const T* q = p + tile_base + static_cast<uint64_t>(lane) * ITEMS;
      // Every path gives each lane the same elements, so the result doesn't depend on alignment.
      if constexpr (kVectorized) {
#pragma unroll
        for (int i = 0; i < ITEMS / VEC; ++i) {
          const auto g = *reinterpret_cast<const GVec<T, VEC>*>(q + i * VEC);
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            x[u][i * VEC + k] = g.v[k];
          }
        }
      } else if constexpr (kDwordAccess<T, ITEMS>) {
        load_dwords<true>(x[u], q, ITEMS, init);
      } else {
#pragma unroll
        for (int k = 0; k < ITEMS; ++k) {
          x[u][k] = q[k];
        }
      }
    }
#pragma unroll
    for (int u = 0; u < kReduceUnroll; ++u) {
      T s = x[u][0];
#pragma unroll
      for (int k = 1; k < ITEMS; ++k) {
        s = binary_op(s, x[u][k]);
      }
#pragma unroll
      for (int offset = 1; offset < kLogicalWarp; offset <<= 1) {
        s = binary_op(s, shfl_down(s, offset, kLogicalWarp));
      }
      if (base + u * tile < end) {
        total = binary_op(total, s);
      }
    }
  }
  return total;
}

// Totals of every chunk except the last one of each row. Requires chunk_size to be a
// multiple of kBlock * ITEMS; warp w folds the w-th of kWarps equal spans of a chunk.
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
    const uint64_t span = chunk_size / kWarps;
    const uint64_t begin = static_cast<uint64_t>(chunk) * chunk_size + warp * span;
    const uint64_t end = begin + span;
    const T* row_src = src + row * row_size;
    const T acc = vec_ok
        ? reduce_span<true, T, ITEMS, VEC>(row_src, begin, end, lane, init, binary_op)
        : reduce_span<false, T, ITEMS, VEC>(row_src, begin, end, lane, init, binary_op);
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

// These kernels are tuned for gfx950 and gfx1250; other targets keep the stock kernels.
inline bool supported_device() {
  static const std::vector<bool> per_device = [] {
    std::vector<bool> v(c10::cuda::device_count());
    auto& hooks = at::detail::getCUDAHooks();
    for (c10::DeviceIndex d = 0; d < static_cast<c10::DeviceIndex>(v.size()); ++d) {
      v[d] = hooks.isGPUArch({"gfx950", "gfx1250", "gfx1250-strict"}, d);
    }
    return v;
  }();
  return per_device[c10::cuda::current_device()];
}

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
  // Vector accesses start at a row start plus a multiple of kItems, so only row starts need
  // kAlign; partial vectors at the end of a row use scalar accesses (1- and 2-byte elements use
  // dword accesses instead; see kDwordAccess).
  const bool vec_ok = static_cast<uintptr_t>(row_size) * sizeof(T) % kAlign == 0 &&
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
  static_assert(Tr::kCheapOp);
  if constexpr (sizeof(T) >= 16) {
    // On 64-lane waves, 16 contiguous 16-byte items per lane spread every load instruction over
    // 64 cache lines; there 2 items are fastest.
    if (at::cuda::warp_size() == 64) {
      return launch_with<2, 1, 2>(tgt, src, num_rows, row_size, init, binary_op);
    }
  }
  launch_with<Tr::kItems, Tr::kVec, Tr::kRun>(tgt, src, num_rows, row_size, init, binary_op);
}

} // namespace inner_scan
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
  if constexpr (inner_scan::Traits<scalar_t, BinaryFunction>::kCheapOp) {
    if (inner_scan::supported_device()) {
      check_fits_in_unsigned(num_rows, "Number of rows (self.numel()/self.size(self.dim()-1))");
      check_fits_in_unsigned(row_size, "row_size");
      inner_scan::launch<scalar_t>(
          result.mutable_data_ptr<scalar_t>(), self.const_data_ptr<scalar_t>(),
          num_rows, row_size, init, binary_op);
      return;
    }
  }
#endif
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
}

template<typename scalar_t, typename BinaryFunction>
void scan_dim(const TensorBase& self, const TensorBase& result,
     int64_t dim, scalar_t init, BinaryFunction binary_op) {
  int ndim = self.dim();
  auto self_ = self.expect_contiguous();
  TORCH_INTERNAL_ASSERT(result.is_contiguous());

  if (self.numel() == self.size(dim)) {
#if defined(USE_ROCM)
    if constexpr (inner_scan::Traits<scalar_t, BinaryFunction>::kCheapOp) {
      // The chunked row scan is as fast as or faster than hipCUB at every length, and deterministic.
      const int64_t n = self.numel();
      if (inner_scan::supported_device() && n <= std::numeric_limits<uint32_t>::max()) {
        inner_scan::launch<scalar_t>(
            result.mutable_data_ptr<scalar_t>(), self_->const_data_ptr<scalar_t>(), 1, n, init, binary_op);
        return;
      }
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
