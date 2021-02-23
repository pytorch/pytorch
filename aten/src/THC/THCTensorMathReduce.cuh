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
