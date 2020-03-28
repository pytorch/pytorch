#ifndef THC_TENSORMATH_REDUCE_CUH
#define THC_TENSORMATH_REDUCE_CUH

#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <THC/THCNumerics.cuh>
#include <THC/THCReduce.cuh>
#include <THC/THCReduceAll.cuh>
#include <THC/THCTensorCopy.hpp>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
#include <thrust/system/cuda/execution_policy.h>
#endif

/*
Reductions that (only) operate on accumulate types.
*/

template <typename T, typename U>
struct WelfordData {
  T mean_;
  T m_2_n_;
  int count_; // do we need int64_t?

  __host__ __device__ WelfordData() {
  }

  // stripping initialization from default constructor to avoid dynamic
  // initialization warning thrown from using this data structure in CUDA kernel
  // as static shared memory.
  __host__ __device__ void reset() {
    mean_ = T(0);
    m_2_n_ = T(0);
    count_ = 0;
  }

  __host__ __device__ WelfordData(const U data_) {
    mean_ = static_cast<T>(data_);
    m_2_n_ = static_cast<T>(0);
    count_ = 1;
  }

  __host__ __device__ WelfordData(const WelfordData &t) :
    mean_(t.mean_),
    m_2_n_(t.m_2_n_),
    count_(t.count_)
  {
  }

  __host__ __device__ WelfordData(const volatile WelfordData &t) :
    mean_(t.mean_),
    m_2_n_(t.m_2_n_),
    count_(t.count_)
  {
  }

  __host__ __device__ volatile WelfordData& operator = (const volatile WelfordData &t) volatile {
    mean_ = t.mean_;
    m_2_n_ = t.m_2_n_;
    count_ = t.count_;
    return *this;
  }

  __host__ __device__ WelfordData& operator = (const WelfordData &t) {
    mean_ = t.mean_;
    m_2_n_ = t.m_2_n_;
    count_ = t.count_;
    return *this;
  }

};


template <typename T>
struct ModifyWelford {
  inline __device__ T operator()(const T &a) const {
    return a;
  }
};

template <typename T, typename U>
struct ReduceWelford {
  inline __device__ WelfordData<T, U> operator()(const WelfordData<T, U> &a, const WelfordData<T, U> &b) const {
    WelfordData<T, U> c;
    c.count_ = THCNumerics<int>::add(a.count_, b.count_);
    T factor = THCNumerics<T>::div(1.0, max(1, c.count_));
    c.mean_ = THCNumerics<T>::mul(THCNumerics<T>::add(THCNumerics<T>::mul(a.mean_, a.count_), THCNumerics<T>::mul(b.mean_, b.count_)), factor);
    c.m_2_n_ = THCNumerics<T>::add(a.m_2_n_, THCNumerics<T>::add(b.m_2_n_, THCNumerics<T>::mul(factor, THCNumerics<T>::mul(a.count_, THCNumerics<T>::mul(b.count_, THCNumerics<T>::pow(THCNumerics<T>::sub(a.mean_, b.mean_), 2) )))));
    return c;
  }
};

template <typename T, typename U>
struct VarianceWelford {
  VarianceWelford(const int _unbiased, const bool _apply_sqrt): unbiased{_unbiased}, apply_sqrt(_apply_sqrt) {}

  inline __device__ T operator()(const WelfordData<T, U> &a) const {
    T res = THCNumerics<T>::div(a.m_2_n_, unbiased ? a.count_ : a.count_-1);
    if (apply_sqrt) {
      return THCNumerics<T>::sqrt(res);
    }
    return res;
  }

  const int unbiased;
  const bool apply_sqrt;
};

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
    return (THCNumerics<T>::gt(a, b) || THCNumerics<T>::isnan(a)) ? a : b;
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

  if (THCNumerics<AccT>::eq(value, scalar_cast<AccT, float>(INFINITY))) {
    // get norm of axis
    for (ptrdiff_t i = tx; i < size; i += step) {
      const AccT val = scalar_cast<AccT>(row[i]);
      buffer[tx] = THCMax<AccT>(buffer[tx], static_cast<AccT>(std::abs(val)));
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
        THCNumerics<AccT>::pow(static_cast<AccT>(std::abs(val)), value)
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
    norm = THCNumerics<AccT>::pow(buffer[0], static_cast<AccT>(1) / value);
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

/*
  Fuses conversions and a TensorDistOp. Needed for Thrust.
*/
template <typename T, typename AccT>
struct ThrustTensorDistOp {
  ThrustTensorDistOp(AccT _exponent) : exponent{_exponent} {}

  __host__ __device__ AccT operator()(T _x, T _y) const {
    const AccT x = scalar_cast<AccT>(_x);
    const AccT y = scalar_cast<AccT>(_y);
    if (THCNumerics<AccT>::eq(exponent, scalar_cast<AccT, float>(0))) {
      const AccT zero = scalar_cast<AccT>(0);
      if (THCNumerics<AccT>::eq(THCNumerics<AccT>::sub(x, y), zero))return zero;
      return scalar_cast<AccT>(1);
    }
    if (THCNumerics<AccT>::eq(exponent, scalar_cast<AccT, float>(1))) {
      return static_cast<AccT>(std::abs(THCNumerics<AccT>::sub(x, y)));
    } else if (THCNumerics<AccT>::eq(exponent, scalar_cast<AccT, float>(2))) {
      return THCNumerics<AccT>::pow(
        THCNumerics<AccT>::sub(x, y), exponent);
    } else {
      return THCNumerics<AccT>::pow(
        static_cast<AccT>(std::abs(THCNumerics<AccT>::sub(x, y))),
        exponent);
    }
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

/* A set of reduction kernels that take in binary ops on thrust pairs (of value, index).
   These are useful when you not only have to do a reduction, but you might have
   to preserve the location of contention (for example min/max operations).
   The structure of the kernels follows the structure of the reduction kernels.
*/
template <typename K, typename Index, class BinaryFunction, typename index_t>
__global__ void
kernelTransformReduceOuterDimIndex(K *tgt1,
                                   Index *tgt2,
                                   K *src_,
                                   index_t num_orows,
                                   index_t num_irows,
                                   index_t row_size,
                                   thrust::pair<K, Index> init,
                                   BinaryFunction binary_op) {
  for (index_t orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (index_t irow = blockIdx.y * blockDim.x + threadIdx.x;
         irow < num_irows;
         irow += gridDim.y * blockDim.x) {
      K *src = src_ + orow * row_size * num_irows + irow;
      thrust::pair<K, Index> acc = init;

      for (index_t col = 0; col < row_size; ++col) {
        // +1 for Lua index
        acc = binary_op(acc,
                        thrust::make_pair<K, Index>(*src, col));
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
  int ndim = THCTensor_nDimensionLegacyAll(state, src);
  int64_t num_orows = 1;
  for (int dim = 0; dim < rdim; dim++) {
    num_orows *= THCTensor_sizeLegacyNoScalars(state, src, dim);
  }
  int64_t row_size = THCTensor_sizeLegacyNoScalars(state, src, rdim);
  int64_t num_irows = 1;
  for (int dim = rdim + 1; dim < ndim; dim++) {
    num_irows *= THCTensor_sizeLegacyNoScalars(state, src, dim);
  }

  int64_t num_threads = std::min(int64_t{512}, num_irows);
  dim3 threads(num_threads);
  int64_t maxGridDim = 1024;
  dim3 grid(std::min(maxGridDim, num_orows),
        std::min(maxGridDim, THCCeilDiv(num_irows, num_threads)));
  auto stream = c10::cuda::getCurrentCUDAStream();

  // Use 32-bit indexing if possible
  if (THCTensor_canUse32BitIndexMath(state, src)) {
    kernelTransformReduceOuterDimIndex
      <<<grid, threads, 0, stream>>>(
        tgt1->template data<ScalarTypeK>(),
        tgt2->template data<ScalarTypeIndex>(),
        src->template data<ScalarTypeK>(),
        static_cast<unsigned>(num_orows),
        static_cast<unsigned>(num_irows),
        static_cast<unsigned>(row_size),
        init, binary_op);
  } else {
    kernelTransformReduceOuterDimIndex
      <<<grid, threads, 0, stream>>>(
        tgt1->template data<ScalarTypeK>(),
        tgt2->template data<ScalarTypeIndex>(),
        src->template data<ScalarTypeK>(),
        num_orows, num_irows, row_size, init, binary_op);
  }

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
template <typename K, typename Index, class BinaryFunction, typename index_t>
__global__ void
kernelTransformReduceInnermostDimIndex(K *tgt1,
                                       Index* tgt2,
                                       K *src_,
                                       index_t num_rows,
                                       index_t row_size,
                                       thrust::pair<K, Index> init,
                                       BinaryFunction binary_op) {
  __shared__ K sbuf[32][16 + 1]; // avoid bank conflict
  __shared__ Index ibuf[32][16 + 1]; // avoid bank conflict

  for (index_t block_row = blockIdx.x * blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    index_t row = block_row + threadIdx.y;
    thrust::pair<K, Index> acc = init;
    if (row < num_rows) {
      K *src = src_ + row * row_size;
      // Sequential reduction within a thread.
      for (index_t col = threadIdx.x; col < row_size; col += blockDim.x) {
        acc = binary_op(acc, thrust::make_pair<K, Index>(src[col], col));
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
  int ndim = THCTensor_nDimensionLegacyAll(state, src);
  int64_t num_rows = 1;
  for (int dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THCTensor_sizeLegacyNoScalars(state, src, dim);
  }
  int64_t row_size = THCTensor_sizeLegacyNoScalars(state, src, ndim - 1);

  dim3 threads(16, 32);
  auto stream = c10::cuda::getCurrentCUDAStream();
  dim3 grid(
    std::min(int64_t{1024}, THCCeilDiv(num_rows, int64_t{threads.y})));

  // Use 32-bit indexing if possible
  if (THCTensor_canUse32BitIndexMath(state, src)) {
    kernelTransformReduceInnermostDimIndex
      <<<grid, threads, 0, stream>>>(
        tgt1->template data<ScalarTypeK>(),
        tgt2->template data<ScalarTypeIndex>(),
        src->template data<ScalarTypeK>(),
        static_cast<unsigned>(num_rows),
        static_cast<unsigned>(row_size),
        init, binary_op);
  } else {
    kernelTransformReduceInnermostDimIndex
      <<<grid, threads, 0, stream>>>(
        tgt1->template data<ScalarTypeK>(),
        tgt2->template data<ScalarTypeIndex>(),
        src->template data<ScalarTypeK>(),
        num_rows, row_size, init, binary_op);
  }

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

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
    return THCNumerics<T>::gt(lhs, rhs) ? lhs : rhs;
  }
};

template <typename T>
struct MinOp {
  __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
    return THCNumerics<T>::lt(lhs, rhs) ? lhs : rhs;
  }
};

#endif // THC_TENSORMATH_REDUCE_CUH
