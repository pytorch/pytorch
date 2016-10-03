#ifndef THC_HALF_AUTO_NUMERICS_INC
#define THC_HALF_AUTO_NUMERICS_INC

#include "THCHalf.h"
#include "THCNumerics.cuh"

// Half numerics functions defined as free functions, so cunn code can be
//written generically, i.e. without calling THCNumerics<half> functions.

#ifdef CUDA_HALF_TENSOR

inline __host__ __device__ half operator+(half a, half b) {
  return THCNumerics<half>::add(a, b);
}

inline __host__ __device__ half operator-(half a) {
  return THCNumerics<half>::neg(a);
}

inline __host__ __device__ half operator-(half a, int b) {
  return THCNumerics<half>::add(a, THCNumerics<half>::neg(ScalarConvert<int, half>::to(b)));
}

inline __host__ __device__ half operator-(int a, half b) {
  return THCNumerics<half>::add(ScalarConvert<int, half>::to(a), THCNumerics<half>::neg(b));
}

// This implementation could move to THCNumerics
inline __host__ __device__ half operator*(half a, half b) {
  #ifdef __CUDA_ARCH__
  #ifdef CUDA_HALF_INSTRUCTIONS
    return __hmul(a, b);
  #else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return __float2half( fa * fb );
  #endif
  #else // __CUDA_ARCH__
    return THC_float2half(THC_half2float(a) * THC_half2float(b));
  #endif
}

inline __host__ __device__ bool operator<(half a, half b) {
  return THCNumerics<half>::lt(a, b);
}

inline __host__ __device__ bool operator<=(half a, half b) {
  return THCNumerics<half>::le(a, b);
}

inline __host__ __device__ bool operator<(half a, int b) {
  return THCNumerics<half>::lt(a, ScalarConvert<int, half>::to(b));
}

inline __host__ __device__ bool operator>(half a, half b) {
  return THCNumerics<half>::gt(a, b);
}

inline __host__ __device__ bool operator>=(half a, half b) {
  return THCNumerics<half>::ge(a, b);
}

inline __host__ __device__ half abs(half a) {
  return THCNumerics<half>::abs(a);
}

inline __host__ __device__ half exp(half a) {
  return THCNumerics<half>::exp(a);
}

inline __host__ __device__ half log1p(half a) {
  return THCNumerics<half>::log1p(a);
}

inline __host__ __device__ half tanh(half a) {
  return THCNumerics<half>::tanh(a);
}

// This implementation could move to THCNumerics
inline __host__ __device__ half operator/(half a, half b) {
  #ifdef __CUDA_ARCH__
  #ifdef CUDA_HALF_INSTRUCTIONS
    return __hdiv(a, b);
  #else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return __float2half( fa / fb );
  #endif
  #else // __CUDA_ARCH__
    return THC_float2half(THC_half2float(a) / THC_half2float(b));
  #endif
}

inline __host__ __device__ half operator/(int a, half b) {
  return ScalarConvert<int, half>::to(a) / b;
}

#endif
#endif
