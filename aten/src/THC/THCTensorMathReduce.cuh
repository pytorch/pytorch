#ifndef THC_TENSORMATH_REDUCE_CUH
#define THC_TENSORMATH_REDUCE_CUH

#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCNumerics.cuh"
#include "THCReduce.cuh"
#include "THCReduceAll.cuh"
#include "THCTensorCopy.hpp"
#include "THCThrustAllocator.cuh"
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

/*
Reductions that (only) operate on accumulate types. 
*/

template <typename T>
struct ReduceAdd {
  inline __device__ T operator()(const T a, const T b) const {
    return THCNumerics<T>::add(a, b);
  }
};

template <typename T>
struct ReduceMultiply {
  inline __device__ T operator()(const T a, const T b) const {
    return THCNumerics<T>::mul(a, b);
  }
};

template <typename T>
struct ReduceDivide {
  ReduceDivide(const T _divisor): divisor{_divisor} {}

  inline __device__ T operator()(const T x) const {
    return THCNumerics<T>::div(x, divisor);
  }

  const T divisor;
};

template <typename T>
struct ReducePow {
  ReducePow(const T _exponent): exponent{_exponent} {}

  inline __device__ T operator()(const T x) const {
    return THCNumerics<T>::pow(x, exponent);
  }

  const T exponent;
};

template <typename T>
struct SquareFunctor {
    SquareFunctor(const T _mean): mean{_mean} {}

    inline __device__ T operator()(const T x) const {
      return THCNumerics<T>::mul(
        THCNumerics<T>::sub(x, mean),
        THCNumerics<T>::sub(x, mean)
        );
    }

    const T mean;
};

template <typename T>
struct ReduceMin {
  inline __device__ T operator()(T a, T b) const {
    return (THCNumerics<T>::lt(a, b) || THCNumerics<T>::isnan(a)) ? a : b;
  }
};

template <typename T>
struct ReduceMax {
  inline __device__ T operator()(T a, T b) const {
#if defined(__HIP_PLATFORM_HCC__)
    return (static_cast<int>(THCNumerics<T>::sub(a, b)) > 0 || THCNumerics<T>::isnan(a)) ? a : b;
#else
    return (THCNumerics<T>::gt(a, b) || THCNumerics<T>::isnan(a)) ? a : b;
#endif
  }
};

struct LogicalAll {
  inline __device__ unsigned char operator()(const unsigned char x,
                                             const unsigned char y) const {
    return (x && y);
  }
};

struct LogicalAny {
  inline __device__ unsigned char operator()(const unsigned char x,
                                             const unsigned char y) const {
    return (x || y);
  }
};

template<typename T>
inline __device__ T THCMax(const T a, const T b) {
  return THCNumerics<T>::gt(a, b) ? a : b;
}

template<typename T, typename AccT>
__global__ void THCTensor_kernel_renorm(T *data, 
                                        const AccT value, 
                                        const ptrdiff_t size, 
                                        const AccT maxnorm) {
  __shared__ AccT buffer[32];
  int64_t tx = threadIdx.x;
  int64_t bx = blockIdx.x;
  int64_t step = blockDim.x;
  T *row = data + size * bx;

  buffer[tx] = scalar_cast<AccT>(0);
  AccT norm;

#if !defined(__HIP_DEVICE_COMPILE__)
  if (THCNumerics<AccT>::eq(value, scalar_cast<AccT, float>(INFINITY))) {
    // get norm of axis
    for (ptrdiff_t i = tx; i < size; i += step) {
      const AccT val = scalar_cast<AccT>(row[i]);
      buffer[tx] = THCMax<AccT>(buffer[tx], THCNumerics<AccT>::abs(val));
    }
    // add (reduce)
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      __syncthreads();
      if (tx < stride)
        buffer[tx] = THCMax<AccT>(buffer[tx], buffer[tx+stride]);
    }
    // clip norms
    __syncthreads();
    norm = buffer[0];
  } else {
    // get norm of axis
    for (ptrdiff_t i = tx; i < size; i += step) {
      const AccT val = scalar_cast<AccT>(row[i]);
      buffer[tx] = THCNumerics<AccT>::add( 
        buffer[tx],
        THCNumerics<AccT>::pow(THCNumerics<AccT>::abs(val), value)
      );
    }
    // add (reduce)
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      __syncthreads();
      if (tx < stride)
        buffer[tx] = THCNumerics<AccT>::add(buffer[tx], buffer[tx+stride]);
    }
    // clip norms
    __syncthreads();
    norm = THCNumerics<AccT>::pow(buffer[0], THCNumerics<AccT>::cinv(value));
  }

  if (THCNumerics<AccT>::gt(norm, maxnorm)) {
    norm = THCNumerics<AccT>::div(
      maxnorm,
      THCNumerics<AccT>::add(norm, scalar_cast<AccT>(1e-7))
    );
    // renormalize
    for (ptrdiff_t i = tx; i < size; i += step) {
      const AccT val = scalar_cast<AccT>(row[i]);
      row[i] = scalar_cast<T>(THCNumerics<AccT>::mul(val, norm));
    }
  }
#endif
}

template <typename T>
struct TensorNonZeroOp {
  TensorNonZeroOp() {}

  __host__ __device__ T operator()(const T lhs) const {
    const T zero = scalar_cast<T>(0);
    if (THCNumerics<T>::eq(lhs, zero)) return zero;
      
    return scalar_cast<T>(1);
  }
};

template <typename T, int StaticExp>
struct TensorNormOp {
  TensorNormOp(T _exponent) : exponent{_exponent} {}

  __host__ __device__ T operator()(const T x) const {
    switch (StaticExp) {
      case 1: return THCNumerics<T>::abs(x);
      case 2: return THCNumerics<T>::mul(x, x);
      default: return THCNumerics<T>::pow(THCNumerics<T>::abs(x), exponent);
    }
  }

  const T exponent;
};

/*
  Fuses conversions and a TensorDistOp. Needed for Thrust.
*/
template <typename T, typename AccT>
struct ThrustTensorDistOp {
  ThrustTensorDistOp(AccT _exponent) : exponent{_exponent} {}

  __host__ __device__ AccT operator()(T _x, T _y) const {
    const AccT x = scalar_cast<AccT>(_x);
    const AccT y = scalar_cast<AccT>(_y);
    return THCNumerics<AccT>::pow(
      THCNumerics<AccT>::abs(THCNumerics<AccT>::sub(x, y)), 
      exponent);
  }

  const AccT exponent;
};

#include <thrust/functional.h>

// Given the sum of values and the sum of squares, compute the variance or standard deviation.
template<typename T, bool flag, bool apply_sqrt>
__forceinline__ __device__ T THCTensor_computeVar(
  T sum, 
  T sum2, 
  const unsigned row_size) {

  T rs2 = scalar_cast<T>(row_size);
  T rs2m = scalar_cast<T>(row_size - 1);
  T zero = scalar_cast<T>(0);

  if (flag) {
    sum = THCNumerics<T>::div(sum, rs2);
    sum2 = THCNumerics<T>::div(sum2, rs2);
    sum2 = THCNumerics<T>::sub(sum2, THCNumerics<T>::mul(sum, sum));
    sum2 = (THCNumerics<T>::lt(sum2, zero) ? zero : sum2);
  } else {
    sum = THCNumerics<T>::div(sum, rs2);
    sum2 = THCNumerics<T>::div(sum2, rs2m);
    sum2 = THCNumerics<T>::sub(sum2,
      THCNumerics<T>::mul(
        THCNumerics<T>::div(rs2 ,rs2m),
        THCNumerics<T>::mul(sum, sum)));
    sum2 = (THCNumerics<T>::lt(sum2, zero) ? zero : sum2);
  }

  if (apply_sqrt)
    return THCNumerics<T>::sqrt(sum2);
  
  return sum2;
}

/* Compute the variance (or standard deviation) along an outer dimension of a tensor.
 *
 * - num_orows is the size of the flattened outer dimensions;
 * - num_irows is the size of the flattened inner dimensions;
 * - row_size is the size of the dimension along which to compute the variance;
 * - if flag is set, normalize by `row_size` instead of `row_size - 1`
 * - if apply_sqrt is set, compute the standard deviation instead of variance
 *
 * The dimensions to the outside and inside of the specified dimension are considered as flattened.
 * Thread blocks with the same blockIdx.y process an "outer row" (i.e. an element of the flattened
 * outer dimensions, which contains several "inner rows").
 * Each thread processes a single inner row at a time.
 */
template<typename T, typename AccT, bool flag, bool apply_sqrt>
__global__ void THCTensor_kernel_varOuterDim(T *tgt, T *src_, unsigned num_orows, unsigned num_irows, unsigned row_size) {
  for (unsigned orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (unsigned irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      T *src = src_ + orow * row_size * num_irows + irow;
      AccT mean = scalar_cast<AccT>(0);
      AccT m2 = scalar_cast<AccT>(0);

      for (unsigned col = 0; col < row_size; ++col) {
        AccT val = scalar_cast<AccT>(*src);
        AccT delta = THCNumerics<AccT>::sub(val, mean);
        mean = THCNumerics<AccT>::add(mean,
            THCNumerics<AccT>::div(delta, scalar_cast<AccT>(col + 1)));
        AccT delta2 = THCNumerics<AccT>::sub(val, mean);
        m2 = THCNumerics<AccT>::add(m2,
            THCNumerics<AccT>::mul(delta, delta2));
        src += num_irows;
      }

      if (flag) {
        m2 = THCNumerics<AccT>::div(m2, scalar_cast<AccT>(row_size));
      } else {
        m2 = THCNumerics<AccT>::div(m2, scalar_cast<AccT>(row_size - 1));
      }

      tgt[orow * num_irows + irow] = scalar_cast<T>(
          apply_sqrt ? THCNumerics<AccT>::sqrt(m2) : m2);
    }
  }
}

template<typename TensorTypeK, typename T, typename AccT, bool apply_sqrt>
__host__ void THCTensor_varOuterDim(THCState *state, TensorTypeK *tgt, TensorTypeK *src, int64_t dimension, int flag) {
  unsigned ndim = THCTensor_nDimensionLegacyAll(state, src);
  // Treat all outer dimensions (i.e. dim < dimension) as one.
  unsigned num_orows = 1;
  for (int64_t dim = 0; dim < dimension; dim++) {
    num_orows *= THCTensor_sizeLegacyNoScalars(state, src, dim);
  }
  unsigned row_size = THCTensor_sizeLegacyNoScalars(state, src, dimension);
  // Treat all inner dimensions (i.e. dim > dimension) as one.
  unsigned num_irows = 1;
  for (unsigned dim = dimension + 1; dim < ndim; dim++) {
    num_irows *= THCTensor_sizeLegacyNoScalars(state, src, dim);
  }

  dim3 threads(min(512, num_irows));
  unsigned maxGridDim = 1024;
  dim3 grid(min(maxGridDim, num_orows), min(maxGridDim, THCCeilDiv(num_irows, threads.x)));

  if (flag) {
    THCTensor_kernel_varOuterDim<T, AccT, true, apply_sqrt><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
        tgt->template data<T>(), src->template data<T>(), num_orows, num_irows, row_size);
  } else {
    THCTensor_kernel_varOuterDim<T, AccT, false, apply_sqrt><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
        tgt->template data<T>(), src->template data<T>(), num_orows, num_irows, row_size);
  }

  cudaError_t errcode = cudaGetLastError();
  if (errcode != cudaSuccess) THError(cudaGetErrorString(errcode));
}

/* Compute the variance (or standard deviation) of the innermost dimension of a tensor.
 *
 * - num_rows is the size of the flattened outer dimensions;
 * - row_size is the size of the innermost dimension;
 * - if flag is set, normalize by `row_size` instead of `row_size - 1`
 * - if apply_sqrt is set, compute the standard deviation instead of variance
 *
 * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is
 * considered as having 'num_rows' rows of size 'row_size'.
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 *
 * Uses Welford's algorithm for numeric stability. Divides the dataset into parallel groups
 * and computes the M2 and mean for each group. (M2 is \sum (x - \bar{x})^2)
 * For example, if the data is split into two groups x and y, the overall M2 can
 * be computed by:
 *
 *    overall_M2 = M2x + nx * (mean(x) - overall_mean)^2
 *               + M2y + ny * (mean(y) - overall_mean)^2
 *
 * This implementation assumes that each block has been launched with 16 x 32 threads.
 */
template<typename T, typename AccT, bool flag, bool apply_sqrt>
__global__ void THCTensor_kernel_varInnermostDim(T *tgt, T *src_, unsigned num_rows, unsigned row_size) {
  /*
   * Each block computes the var/std of blockDim.y (32) rows at once.
   * One can visualize the computation as a 16 (x) by 32 (y) grid.
   * - Each of the 32 rows of the block is responsible for the computation
   *   of one input row.
   * - Each row has 16 columns; the variance computation of one input row is
   *   split between 16 threads.
   * - Each of those 16 threads handles the accumulation of 1/16 of the input
   *   row's data.
   */
  for (unsigned block_row = blockIdx.x * blockDim.y; block_row < num_rows; block_row += blockDim.y * gridDim.x) {
    unsigned row = block_row + threadIdx.y;

    /*
     * Compute local mean, local M2 via Welford's algorithm for this thread.
     */
    AccT acc_zero = scalar_cast<AccT>(0);
    AccT local_mean = acc_zero;
    AccT local_M2 = acc_zero;
    unsigned count = 0;

    if (row < num_rows) {
      T *src = src_ + row * row_size;

      for (unsigned col = threadIdx.x; col < row_size; col += blockDim.x) {
        ++count;
        AccT val = scalar_cast<AccT>(src[col]);
        AccT delta = THCNumerics<AccT>::sub(val, local_mean);
        local_mean = THCNumerics<AccT>::add(
            local_mean,
            THCNumerics<AccT>::div(delta, scalar_cast<AccT>(count)));
        AccT delta2 = THCNumerics<AccT>::sub(val, local_mean);
        local_M2 = THCNumerics<AccT>::add(
            local_M2,
            THCNumerics<AccT>::mul(delta, delta2));
      }
    }

    AccT local_sum =
        THCNumerics<AccT>::mul(local_mean, scalar_cast<AccT>(count));

    /*
     * We are reducing across each row of 16 threads to find the true sum of the
     * entire input row. The warp shfl xor loop ultimately gives each thread the
     * true sum.
     */
    for (unsigned lane_mask = 8; lane_mask > 0; lane_mask >>= 1) {
      local_sum = THCNumerics<AccT>::add(local_sum, 
          WARP_SHFL_XOR((row < num_rows) ? local_sum : acc_zero, lane_mask, 16));
    }
    AccT true_mean = THCNumerics<AccT>::div(local_sum, 
      scalar_cast<AccT>(row_size));

    /*
     * Adjust each local_M2 according to the following:
     *   adjusted_M2 = local_M2 + mean_diff * mean_diff * count
     * The sum of these adjusted M2s is equal to the overall M2.
     */
    AccT adjusted_M2 = acc_zero;
    if (row < num_rows) {
      AccT mean_diff = THCNumerics<AccT>::sub(true_mean, local_mean);
      adjusted_M2 = THCNumerics<AccT>::add(
          local_M2,
          THCNumerics<AccT>::mul(
              THCNumerics<AccT>::mul(mean_diff, mean_diff),
              scalar_cast<AccT>(count)));
    }

    /*
     * Sums the adjusted M2s. The thread with threadIdx.x == 0 has
     * the total sum, which is equal to the M2 for the entire input row.
     */
    for (unsigned s = 8; s >= 1; s >>= 1) {
      adjusted_M2 = THCNumerics<AccT>::add(adjusted_M2, 
          WARP_SHFL_DOWN((row < num_rows) ? adjusted_M2 : acc_zero, s, 16));
    }

    if (row < num_rows && threadIdx.x == 0) {
      AccT M2 = adjusted_M2;
      AccT variance;
      if (flag) {
        variance = THCNumerics<AccT>::div(M2, scalar_cast<AccT>(row_size));
      } else {
        variance = THCNumerics<AccT>::div(M2, scalar_cast<AccT>(row_size - 1));
      }
      tgt[row] = scalar_cast<T>(
          apply_sqrt ? THCNumerics<AccT>::sqrt(variance) : variance);
    }
  }
}

template<typename TensorTypeK, typename T, typename AccT, bool apply_sqrt>
__host__ void THCTensor_varInnermostDim(THCState *state, TensorTypeK *tgt, TensorTypeK *src, int flag) {
  unsigned ndim = THCTensor_nDimensionLegacyAll(state, src);
  // Treat all outer dimensions as a single dimension.
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THCTensor_sizeLegacyNoScalars(state, src, dim);
  }
  unsigned row_size = THCTensor_sizeLegacyNoScalars(state, src, ndim - 1);

  // From limited testing, 16x32 seemed a good compromise for handling both long and short dimensions.
  dim3 threads(16, 32);
  dim3 grid(min(1024, THCCeilDiv(num_rows, threads.y)));

  if (flag) {
    THCTensor_kernel_varInnermostDim<T, AccT, true, apply_sqrt><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
        tgt->template data<T>(), src->template data<T>(), num_rows, row_size);
  } else {
    THCTensor_kernel_varInnermostDim<T, AccT, false, apply_sqrt><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
        tgt->template data<T>(), src->template data<T>(), num_rows, row_size);
  }

  cudaError_t errcode = cudaGetLastError();
  if (errcode != cudaSuccess) THError(cudaGetErrorString(errcode));
}


/* A set of reduction kernels that take in binary ops on thrust pairs (of value, index).
   These are useful when you not only have to do a reduction, but you might have
   to preserve the location of contention (for example min/max operations).
   The structure of the kernels follows the structure of the reduction kernels.
*/
template <typename K, typename Index, class BinaryFunction>
__global__ void
kernelTransformReduceOuterDimIndex(K *tgt1,
                                   Index *tgt2,
                                   K *src_,
                                   unsigned num_orows,
                                   unsigned num_irows,
                                   unsigned row_size,
                                   thrust::pair<K, Index> init,
                                   BinaryFunction binary_op) {
  for (unsigned orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (unsigned irow = blockIdx.y * blockDim.x + threadIdx.x;
         irow < num_irows;
         irow += gridDim.y * blockDim.x) {
      K *src = src_ + orow * row_size * num_irows + irow;
      thrust::pair<K, Index> acc = init;

      for (unsigned col = 0; col < row_size; ++col) {
        // +1 for Lua index
        acc = binary_op(acc,
                        thrust::make_pair<K, Index>(*src, col + TH_INDEX_BASE));
        src += num_irows;
      }

      tgt1[orow * num_irows + irow] = acc.first;
      tgt2[orow * num_irows + irow] = acc.second;
    }
  }
}

template <typename ScalarTypeK,
          typename ScalarTypeIndex,
          typename TensorTypeK,
          typename TensorTypeIndex,
          typename BinaryFunction>
__host__ void
THC_transformReduceOuterDimIndex(THCState *state,
                                 TensorTypeK *tgt1,
                                 TensorTypeIndex *tgt2,
                                 TensorTypeK *src,
                                 int64_t rdim,
                                 const thrust::pair<ScalarTypeK, ScalarTypeIndex>& init,
                                 BinaryFunction binary_op) {
  unsigned ndim = THCTensor_nDimensionLegacyAll(state, src);
  unsigned num_orows = 1;
  for (int64_t dim = 0; dim < rdim; dim++) {
    num_orows *= THCTensor_sizeLegacyNoScalars(state, src, dim);
  }
  unsigned row_size = THCTensor_sizeLegacyNoScalars(state, src, rdim);
  unsigned num_irows = 1;
  for (unsigned dim = rdim + 1; dim < ndim; dim++) {
    num_irows *= THCTensor_sizeLegacyNoScalars(state, src, dim);
  }

  dim3 threads(min(512, num_irows));
  unsigned maxGridDim = 1024;
  dim3 grid(min(maxGridDim, num_orows),
            min(maxGridDim, THCCeilDiv(num_irows, threads.x)));

  kernelTransformReduceOuterDimIndex
    <<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
      tgt1->template data<ScalarTypeK>(),
      tgt2->template data<ScalarTypeIndex>(),
      src->template data<ScalarTypeK>(),
      num_orows, num_irows, row_size, init, binary_op);

  THCudaCheck(cudaGetLastError());
}

/* Reduce the innermost dimension of a tensor (on thrust::pair functors which are (value, index))
 *
 * For an n-d tensor (n <= 4) where the reduction is along the innermost dimension:
 *
 * - block.x is the innermost dimension, i.e. dimension 0;
 * - block.y and grid.y make up dimension 1; and
 * - grid.x and grid z are the remaining two outer dimensions (if any)
 *
 * Reduction along other dimensions is handled in a separate kernel.
 */
template <typename K, typename Index, class BinaryFunction>
__global__ void
kernelTransformReduceInnermostDimIndex(K *tgt1,
                                       Index* tgt2,
                                       K *src_,
                                       unsigned num_rows,
                                       unsigned row_size,
                                       thrust::pair<K, Index> init,
                                       BinaryFunction binary_op) {
  __shared__ K sbuf[32][16 + 1]; // avoid bank conflict
  __shared__ Index ibuf[32][16 + 1]; // avoid bank conflict

  for (unsigned block_row = blockIdx.x * blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    unsigned row = block_row + threadIdx.y;
    thrust::pair<K, Index> acc = init;
    if (row < num_rows) {
      K *src = src_ + row * row_size;
      // Sequential reduction within a thread.
      for (unsigned col = threadIdx.x; col < row_size; col += blockDim.x) {
        acc = binary_op(acc, thrust::make_pair<K, Index>(src[col], col + TH_INDEX_BASE));
      }
    }

    sbuf[threadIdx.y][threadIdx.x] = acc.first;
    ibuf[threadIdx.y][threadIdx.x] = acc.second;

    __syncthreads();

    // Reduce intermediate values to single value.
    K* sline = &sbuf[threadIdx.y][0];
    Index* iline = &ibuf[threadIdx.y][0];
    for (unsigned s = 8; s > 0; s >>= 1) {
      if (row < num_rows && threadIdx.x < s) {
        thrust::pair<K, Index> arg1 =
          thrust::make_pair<K, Index>(sline[threadIdx.x], iline[threadIdx.x]);
        thrust::pair<K, Index> arg2 =
          thrust::make_pair<K, Index>(sline[threadIdx.x + s], iline[threadIdx.x + s]);
        thrust::pair<K, Index> res = binary_op(arg1, arg2);

        sline[threadIdx.x] = res.first;
        iline[threadIdx.x] = res.second;
      }
      __syncthreads();
    }

    if (row < num_rows && threadIdx.x == 0) {
      tgt1[row] = sline[0];
      tgt2[row] = iline[0];
    }
    __syncthreads();
  }
}

template <typename ScalarTypeK,
          typename ScalarTypeIndex,
          typename TensorTypeK,
          typename TensorTypeIndex,
          typename BinaryFunction>
__host__ void
THC_transformReduceInnermostDimIndex(THCState *state,
                                     TensorTypeK *tgt1,
                                     TensorTypeIndex *tgt2,
                                     TensorTypeK *src,
                                     const thrust::pair<ScalarTypeK, ScalarTypeIndex>& init,
                                     BinaryFunction binary_op) {
  unsigned ndim = THCTensor_nDimensionLegacyAll(state, src);
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THCTensor_sizeLegacyNoScalars(state, src, dim);
  }
  unsigned row_size = THCTensor_sizeLegacyNoScalars(state, src, ndim - 1);

  dim3 threads(16, 32);
  dim3 grid(min(1024, THCCeilDiv(num_rows, threads.y)));

  kernelTransformReduceInnermostDimIndex
    <<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
      tgt1->template data<ScalarTypeK>(),
      tgt2->template data<ScalarTypeIndex>(),
      src->template data<ScalarTypeK>(),
      num_rows, row_size, init, binary_op);

  THCudaCheck(cudaGetLastError());
}

template <typename ScalarTypeK,
          typename ScalarTypeIndex,
          typename TensorTypeK,
          typename TensorTypeIndex,
          typename BinaryFunction>
void
THC_reduceDimIndex(THCState *state,
                   TensorTypeK *tgt1_,
                   TensorTypeIndex *tgt2_,
                   TensorTypeK *src,
                   int64_t dimension,
                   int keepdim,
                   const thrust::pair<ScalarTypeK, ScalarTypeIndex>& init,
                   BinaryFunction binary_op)
{
  THArgCheck(dimension >= 0 &&
             dimension < THCTensor_nDimensionLegacyAll(state, src),
             3, "dimension out of range");


  // Unsqueeze tgt1_/tgt_2 if necessary so that their contiguity traits
  // are preserved if they are the same size as the correct reduction output.
  int src_dims = THCTensor_nDimensionLegacyAll(state, src);
  THCTensor_preserveReduceDimSemantics(
      state, tgt1_, src_dims, dimension, keepdim);
  THCTensor_preserveReduceDimSemantics(
      state, tgt2_, src_dims, dimension, keepdim);

  std::vector<int64_t> dim = THTensor_sizesLegacyNoScalars(src);
  dim[dimension] = 1;
  THCTensor_resize(state, tgt1_, dim, {});
  THCTensor_resize(state, tgt2_, dim, {});

  TensorTypeK *tgt1 = (TensorTypeK*)THCTensor_newContiguous<ScalarTypeK>(state, tgt1_);
  TensorTypeIndex *tgt2 = (TensorTypeIndex*)THCTensor_newContiguous<ScalarTypeIndex>(state, tgt2_);
  src = (TensorTypeK*)THCTensor_newContiguous<ScalarTypeK>(state, src);

  if (dimension == THCTensor_nDimensionLegacyAll(state, src) - 1) {
    THC_transformReduceInnermostDimIndex(state, tgt1, tgt2, src, init, binary_op);
  } else {
    THC_transformReduceOuterDimIndex(state, tgt1, tgt2, src, dimension, init, binary_op);
  }

  THCTensor_free(state, src);
  THCTensor_freeCopyTo<ScalarTypeK>(state, tgt1, tgt1_);
  THCTensor_freeCopyTo<ScalarTypeIndex>(state, tgt2, tgt2_);
  if (!keepdim) {
    THCTensor_squeeze1d(state, tgt1_, tgt1_, dimension);
    THCTensor_squeeze1d(state, tgt2_, tgt2_, dimension);
  }
}

template <typename T, typename Index>
struct MaxValuePair {
  __host__ __device__
  thrust::pair<T, Index> operator()(const thrust::pair<T, Index>& a,
                                    const thrust::pair<T, Index>& b) {
    return (THCNumerics<T>::ge(a.first, b.first) ||
            THCNumerics<T>::isnan(a.first)) ? a : b;
  }
};

template <typename T, typename Index>
struct MinValuePair {
  __host__ __device__
  thrust::pair<T, Index> operator()(const thrust::pair<T, Index>& a,
                                    const thrust::pair<T, Index>& b) {
    return (THCNumerics<T>::le(a.first, b.first) ||
            THCNumerics<T>::isnan(a.first)) ? a : b;
  }
};

template <typename T>
struct AddOp {
  __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
    return THCNumerics<T>::add(lhs, rhs);
  }
};

template <typename T>
struct MulOp {
  __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
    return THCNumerics<T>::mul(lhs, rhs);
  }
};

#endif // THC_TENSORMATH_REDUCE_CUH
