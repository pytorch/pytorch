#ifndef THC_TENSORMATH_REDUCE_CUH
#define THC_TENSORMATH_REDUCE_CUH

#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCNumerics.cuh"
#include "THCReduce.cuh"
#include "THCReduceAll.cuh"
#include <thrust/functional.h>

// Reduction operators that support `half`, unlike Thrust
template <typename InT, typename AccT>
struct ReduceAdd {
  inline __device__ AccT operator()(AccT a, InT b) const {
    return a + (AccT) b;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct ReduceAdd<half, half> {
  inline __device__ half operator()(half a, half b) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hadd(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return __float2half(fa + fb);
#endif
  }
};

template <>
struct ReduceAdd<half, float> {
  inline __device__ float operator()(float a, half b) const {
    return a + __half2float(b);
  }
};
#endif // CUDA_HALF_TENSOR

template <typename InT, typename AccT>
struct ReduceMultiply {
  inline __device__ AccT operator()(AccT a, InT b) const {
    return a * (AccT) b;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct ReduceMultiply<half, half> {
  inline __device__ half operator()(half a, half b) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hmul(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return __float2half(fa * fb);
#endif
  }
};

template <>
struct ReduceMultiply<half, float> {
  inline __device__ float operator()(float a, half b) const {
    return a * __half2float(b);
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct ReduceMin {
  inline __device__ T operator()(T a, T b) const {
    return THCNumerics<T>::lt(a, b) ? a : b;
  }
};

template <typename T>
struct ReduceMax {
  inline __device__ T operator()(T a, T b) const {
    return THCNumerics<T>::gt(a, b) ? a : b;
  }
};

struct LogicalAll {
  inline __device__ unsigned char operator()(unsigned char x,
                                             unsigned char y) const {
    return (x && y);
  }
};

struct LogicalAny {
  inline __device__ unsigned char operator()(unsigned char x,
                                             unsigned char y) const {
    return (x || y);
  }
};


#include <thrust/functional.h>

// Given the sum of values and the sum of squares, compute the variance or standard deviation.
template<typename Real, bool flag, bool apply_sqrt>
__forceinline__ __device__ Real THCTensor_computeVar(Real sum, Real sum2, unsigned row_size) {
  if (flag) {
    sum /= row_size;
    sum2 /= row_size;
    sum2 -= sum * sum;
    sum2 = (sum2 < 0 ? 0 : sum2);
  }
  else {
    sum /= row_size;
    sum2 /= row_size - 1;
    sum2 -= ((Real)row_size) / ((Real)(row_size - 1)) * sum * sum;
    sum2 = (sum2 < 0 ? 0 : sum2);
  }
  if (apply_sqrt)
    return sqrt(sum2);
  else
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
template<typename Real, bool flag, bool apply_sqrt>
__global__ void THCTensor_kernel_varOuterDim(Real *tgt, Real *src_, unsigned num_orows, unsigned num_irows, unsigned row_size)
{
  for (unsigned orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (unsigned irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      Real *src = src_ + orow * row_size * num_irows + irow;
      Real sum = 0, sum2 = 0;

      for (unsigned col = 0; col < row_size; ++col) {
        Real val = *src;
        sum += val;
        sum2 += val * val;

        src += num_irows;
      }

      tgt[orow * num_irows + irow] = THCTensor_computeVar<Real, flag, apply_sqrt>(sum, sum2, row_size);
    }
  }
}

template<typename TensorTypeK, bool apply_sqrt>
__host__ void THCTensor_varOuterDim(THCState *state, TensorTypeK *tgt, TensorTypeK *src, long dimension, int flag)
{
  unsigned ndim = TensorUtils<TensorTypeK>::getDims(state, src);
  // Treat all outer dimensions (i.e. dim < dimension) as one.
  unsigned num_orows = 1;
  for (unsigned dim = 0; dim < dimension; dim++) {
    num_orows *= TensorUtils<TensorTypeK>::getSize(state, src, dim);
  }
  unsigned row_size = TensorUtils<TensorTypeK>::getSize(state, src, dimension);
  // Treat all inner dimensions (i.e. dim > dimension) as one.
  unsigned num_irows = 1;
  for (unsigned dim = dimension + 1; dim < ndim; dim++) {
    num_irows *= TensorUtils<TensorTypeK>::getSize(state, src, dim);
  }

  dim3 threads(min(512, num_irows));
  unsigned maxGridDim = 1024;
  dim3 grid(min(maxGridDim, num_orows), min(maxGridDim, THCCeilDiv(num_irows, threads.x)));

  if (flag) {
    THCTensor_kernel_varOuterDim<float, true, apply_sqrt><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
        TensorUtils<TensorTypeK>::getData(state, tgt), TensorUtils<TensorTypeK>::getData(state, src), num_orows, num_irows, row_size);
  } else {
    THCTensor_kernel_varOuterDim<float, false, apply_sqrt><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
        TensorUtils<TensorTypeK>::getData(state, tgt), TensorUtils<TensorTypeK>::getData(state, src), num_orows, num_irows, row_size);
  }
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
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
 */
template<typename Real, bool flag, bool apply_sqrt>
__global__ void THCTensor_kernel_varInnermostDim(Real *tgt, Real *src_, unsigned num_rows, unsigned row_size)
{
  __shared__ Real ssum[32][16];
  __shared__ Real ssum2[32][16];

  for (unsigned block_row = blockIdx.x * blockDim.y; block_row < num_rows; block_row += blockDim.y * gridDim.x) {
    unsigned row = block_row + threadIdx.y;
    Real sum = 0, sum2 = 0;
    if (row < num_rows) {
      Real *src = src_ + row * row_size;
      // Sequential reduction within a thread.
      for (unsigned col = threadIdx.x; col < row_size; col += blockDim.x) {
        Real val = src[col];
        sum += val;
        sum2 += val * val;
      }
    }
    ssum[threadIdx.y][threadIdx.x] = sum;
    ssum2[threadIdx.y][threadIdx.x] = sum2;
    __syncthreads();

    // Reduce intermediate values to single value.
    for (unsigned s = 8; s > 1; s >>= 1) {
      if (row < num_rows && threadIdx.x < s) {
        ssum[threadIdx.y][threadIdx.x] += ssum[threadIdx.y][threadIdx.x + s];
        ssum2[threadIdx.y][threadIdx.x] += ssum2[threadIdx.y][threadIdx.x + s];
      }
      __syncthreads();
    }

    if (row < num_rows && threadIdx.x == 0) {
      sum = ssum[threadIdx.y][0] + ssum[threadIdx.y][1];
      sum2 = ssum2[threadIdx.y][0] + ssum2[threadIdx.y][1];
      tgt[row] = THCTensor_computeVar<Real, flag, apply_sqrt>(sum, sum2, row_size);
    }
    __syncthreads();
  }
}

template<typename TensorTypeK, bool apply_sqrt>
__host__ void THCTensor_varInnermostDim(THCState *state, TensorTypeK *tgt, TensorTypeK *src, int flag)
{
  unsigned ndim = TensorUtils<TensorTypeK>::getDims(state, src);
  // Treat all outer dimensions as a single dimension.
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= TensorUtils<TensorTypeK>::getSize(state, src, dim);
  }
  unsigned row_size = TensorUtils<TensorTypeK>::getSize(state, src, ndim - 1);

  // From limited testing, 16x32 seemed a good compromise for handling both long and short dimensions.
  dim3 threads(16, 32);
  dim3 grid(min(1024, THCCeilDiv(num_rows, threads.y)));

  if (flag) {
    THCTensor_kernel_varInnermostDim<float, true, apply_sqrt><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
        TensorUtils<TensorTypeK>::getData(state, tgt), TensorUtils<TensorTypeK>::getData(state, src), num_rows, row_size);
  } else {
    THCTensor_kernel_varInnermostDim<float, false, apply_sqrt><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
        TensorUtils<TensorTypeK>::getData(state, tgt), TensorUtils<TensorTypeK>::getData(state, src), num_rows, row_size);
  }
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
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
        acc = binary_op(thrust::make_pair<K, Index>(*src, col + TH_INDEX_BASE),
                        acc);
        src += num_irows;
      }

      tgt1[orow * num_irows + irow] = acc.first;
      tgt2[orow * num_irows + irow] = acc.second;
    }
  }
}

template <typename TensorTypeK,
          typename TensorTypeIndex,
          typename BinaryFunction>
__host__ void
THC_transformReduceOuterDimIndex(THCState *state,
                                 TensorTypeK *tgt1,
                                 TensorTypeIndex *tgt2,
                                 TensorTypeK *src,
                                 long rdim,
                                 const thrust::pair<
                                 typename TensorUtils<TensorTypeK>::DataType,
                                 typename TensorUtils<TensorTypeIndex>::DataType>& init,
                                 BinaryFunction binary_op) {
  unsigned ndim = TensorUtils<TensorTypeK>::getDims(state, src);
  unsigned num_orows = 1;
  for (long dim = 0; dim < rdim; dim++) {
    num_orows *= TensorUtils<TensorTypeK>::getSize(state, src, dim);
  }
  unsigned row_size = TensorUtils<TensorTypeK>::getSize(state, src, rdim);
  unsigned num_irows = 1;
  for (unsigned dim = rdim + 1; dim < ndim; dim++) {
    num_irows *= TensorUtils<TensorTypeK>::getSize(state, src, dim);
  }

  dim3 threads(min(512, num_irows));
  unsigned maxGridDim = 1024;
  dim3 grid(min(maxGridDim, num_orows),
            min(maxGridDim, THCCeilDiv(num_irows, threads.x)));

  kernelTransformReduceOuterDimIndex
    <<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
      TensorUtils<TensorTypeK>::getData(state, tgt1),
      TensorUtils<TensorTypeIndex>::getData(state, tgt2),
      TensorUtils<TensorTypeK>::getData(state, src),
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
        acc = binary_op(thrust::make_pair<K, Index>(src[col], col + TH_INDEX_BASE), acc);
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

template <typename TensorTypeK,
          typename TensorTypeIndex,
          typename BinaryFunction>
__host__ void
THC_transformReduceInnermostDimIndex(THCState *state,
                                     TensorTypeK *tgt1,
                                     TensorTypeIndex *tgt2,
                                     TensorTypeK *src,
                                     const thrust::pair<
                                     typename TensorUtils<TensorTypeK>::DataType,
                                     typename TensorUtils<TensorTypeIndex>::DataType>& init,
                                     BinaryFunction binary_op) {
  unsigned ndim = TensorUtils<TensorTypeK>::getDims(state, src);
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= TensorUtils<TensorTypeK>::getSize(state, src, dim);
  }
  unsigned row_size = TensorUtils<TensorTypeK>::getSize(state, src, ndim - 1);

  dim3 threads(16, 32);
  dim3 grid(min(1024, THCCeilDiv(num_rows, threads.y)));

  kernelTransformReduceInnermostDimIndex
    <<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
      TensorUtils<TensorTypeK>::getData(state, tgt1),
      TensorUtils<TensorTypeIndex>::getData(state, tgt2),
      TensorUtils<TensorTypeK>::getData(state, src),
      num_rows, row_size, init, binary_op);

  THCudaCheck(cudaGetLastError());
}

template <typename TensorTypeK,
          typename TensorTypeIndex,
          typename BinaryFunction>
void
THC_reduceDimIndex(THCState *state,
                   TensorTypeK *tgt1_,
                   TensorTypeIndex *tgt2_,
                   TensorTypeK *src,
                   long dimension,
                   const thrust::pair<
                   typename TensorUtils<TensorTypeK>::DataType,
                   typename TensorUtils<TensorTypeIndex>::DataType>& init,
                   BinaryFunction binary_op)
{
  THArgCheck(dimension >= 0 &&
             dimension < TensorUtils<TensorTypeK>::getDims(state, src),
             3, "dimension out of range");

  THLongStorage *dim = TensorUtils<TensorTypeK>::newSizeOf(state, src);
  THLongStorage_set(dim, dimension, 1);
  TensorUtils<TensorTypeK>::resize(state, tgt1_, dim, NULL);
  TensorUtils<TensorTypeIndex>::resize(state, tgt2_, dim, NULL);
  THLongStorage_free(dim);

  TensorTypeK *tgt1 = TensorUtils<TensorTypeK>::newContiguous(state, tgt1_);
  TensorTypeIndex *tgt2 = TensorUtils<TensorTypeIndex>::newContiguous(state, tgt2_);
  src = TensorUtils<TensorTypeK>::newContiguous(state, src);

  if (dimension == TensorUtils<TensorTypeK>::getDims(state, src) - 1) {
    THC_transformReduceInnermostDimIndex(state, tgt1, tgt2, src, init, binary_op);
  } else {
    THC_transformReduceOuterDimIndex(state, tgt1, tgt2, src, dimension, init, binary_op);
  }

  TensorUtils<TensorTypeK>::free(state, src);
  TensorUtils<TensorTypeK>::freeCopyTo(state, tgt1, tgt1_);
  TensorUtils<TensorTypeIndex>::freeCopyTo(state, tgt2, tgt2_);
}

template <typename T, typename Index>
struct MaxValuePair {
  __host__ __device__
  thrust::pair<T, Index> operator()(const thrust::pair<T, Index>& a,
                                    const thrust::pair<T, Index>& b) {
    return THCNumerics<T>::ge(a.first, b.first) ? a : b;
  }
};

template <typename T, typename Index>
struct MinValuePair {
  __host__ __device__
  thrust::pair<T, Index> operator()(const thrust::pair<T, Index>& a,
                                    const thrust::pair<T, Index>& b) {
    return THCNumerics<T>::le(a.first, b.first) ? a : b;
  }
};

#endif // THC_TENSORMATH_REDUCE_CUH
