#ifndef THC_TENSORMATH_REDUCE_CUH
#define THC_TENSORMATH_REDUCE_CUH

#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <THC/THCNumerics.cuh>
#include <THC/THCReduce.cuh>
#include <THC/THCTensorCopy.hpp>

/*
Reductions that (only) operate on accumulate types.
*/

template <typename T>
struct ReduceAdd {
  inline __device__ T operator()(const T a, const T b) const {
    return THCNumerics<T>::add(a, b);
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
