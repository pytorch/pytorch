#ifndef THC_TENSORMATH_POINTWISE_CUH
#define THC_TENSORMATH_POINTWISE_CUH

#include <type_traits>
#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <TH/THHalf.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCApply.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCReduce.cuh>

template <typename T>
struct TensorCAddOp {
  TensorCAddOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out += val * *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = *in1 + val * *in2;
  }

  T val;
};

template <typename T>
struct TensorMulOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out *= *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = *in1 * *in2;
  }
};

template <typename T>
struct TensorCFmodOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *out % *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = *in1 % *in2;
  }
};

template <>
struct TensorCFmodOp<float> {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = fmodf(*out, *in);
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = fmodf(*in1, *in2);
  }
};

template <>
struct TensorCFmodOp<double> {
  __device__ __forceinline__ void operator()(double* out, double* in) {
    *out = fmod(*out, *in);
  }

  __device__ __forceinline__ void operator()(double* out, double* in1, double* in2) {
    *out = fmod(*in1, *in2);
  }
};

template <>
struct TensorCFmodOp<at::Half> {
  __device__ __forceinline__ void operator()(at::Half* out, at::Half* in) {
    *out = fmodf(*out, *in);
  }

  __device__ __forceinline__ void operator()(at::Half* out, at::Half* in1, at::Half* in2) {
    *out = fmodf(*in1, *in2);
  }
};

template <typename T>
struct TensorClampOp {
  TensorClampOp(T min, T max) : minValue(min), maxValue(max) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    T val = THCNumerics<T>::lt(*in, minValue) ? minValue : *in;
    *out = THCNumerics<T>::gt(val, maxValue) ? maxValue : val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    T val = THCNumerics<T>::lt(*v, minValue) ? minValue : *v;
    *v = THCNumerics<T>::gt(val, maxValue) ? maxValue : val;
  }

  const T minValue;
  const T maxValue;
};

template <typename T>
struct TensorCrossOp {
  TensorCrossOp(int64_t sx, int64_t sy, int64_t so) : sx(sx), sy(sy), so(so) {}

  __device__ __forceinline__ void operator()(T* out, T* x, T*y) {
    T val0 = THCNumerics<T>::sub(
        THCNumerics<T>::mul(x[1 * sx], y[2 * sy]),
        THCNumerics<T>::mul(x[2 * sx], y[1 * sy])
    );

    T val1 = THCNumerics<T>::sub(
        THCNumerics<T>::mul(x[2 * sx], y[0 * sy]),
        THCNumerics<T>::mul(x[0 * sx], y[2 * sy])
    );

    T val2 = THCNumerics<T>::sub(
        THCNumerics<T>::mul(x[0 * sx], y[1 * sy]),
        THCNumerics<T>::mul(x[1 * sx], y[0 * sy])
    );

    out[0 * so] = val0;
    out[1 * so] = val1;
    out[2 * so] = val2;
  }

  const int64_t sx, sy, so;
};

template <typename T>
struct TensorMaxOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = THCNumerics<T>::gt(*out, *in) ? *out : *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = THCNumerics<T>::gt(*in1, *in2) ? *in1 : *in2;
  }
};

template <typename T>
struct TensorMinOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = THCNumerics<T>::lt(*out, *in) ? *out : *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = THCNumerics<T>::lt(*in1, *in2) ? *in1 : *in2;
  }
};

template <typename T>
struct TensorMaxValueOp {
  TensorMaxValueOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out) {
    *out = THCNumerics<T>::lt(*out, val) ? val : *out;  // this order propagates NaN
  }

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = THCNumerics<T>::lt(*in, val) ? val : *in;  // this order propagates NaN
  }

  T val;
};

template <typename T>
struct TensorMinValueOp {
  TensorMinValueOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out) {
    *out = THCNumerics<T>::gt(*out, val) ? val : *out;  // this order propagates NaN
  }

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = THCNumerics<T>::gt(*in, val) ? val : *in;  // this order propagates NaN
  }

  T val;
};

#endif // THC_TENSORMATH_POINTWISE_CUH
