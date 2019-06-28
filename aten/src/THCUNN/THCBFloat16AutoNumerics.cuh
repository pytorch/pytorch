#ifndef THC_BFloat16_AUTO_NUMERICS_INC
#define THC_BFloat16_AUTO_NUMERICS_INC

#include <c10/util/BFloat16.h>
#include <THC/THCNumerics.cuh>

// WARNING: THCNumerics is being deprecated. Read the comments and function usage
//          in THCNumerics to learn about the deprecation
//
// BFloat16 numerics functions defined as free functions, so cunn code can be
// written generically, i.e. without excessive calling of THCNumerics<THBFloat16> functions.

// these functions should move to THCNumerics

inline __host__ __device__ BFloat16 fmaxType(BFloat16 x, BFloat16 y) {
  return THCNumerics<BFloat16>::ge(x, y) ? x : y;
}

inline __host__ __device__ float fmaxType(float x, BFloat16 y) {
  return fmaxf(x, ScalarConvert<BFloat16, float>::to(y));
}

inline __host__ __device__ float fmaxType(float x, float y) {
  return fmaxf(x, y);
}

inline __host__ __device__ double fmaxType(double x, double y) {
  return fmax(x, y);
}


// arithmetic functions

inline __host__ __device__ BFloat16 abs(BFloat16 a) {
  return THCNumerics<BFloat16>::abs(a);
}

inline __host__ __device__ BFloat16 exp(BFloat16 a) {
  return THCNumerics<BFloat16>::exp(a);
}

inline __host__ __device__ BFloat16 log10(BFloat16 a) {
  return THCNumerics<BFloat16>::log10(a);
}

inline __host__ __device__ BFloat16 log1p(BFloat16 a) {
  return THCNumerics<BFloat16>::log1p(a);
}

inline __host__ __device__ BFloat16 log2(BFloat16 a) {
  return THCNumerics<BFloat16>::log2(a);
}

inline __host__ __device__ BFloat16 expm1(BFloat16 a) {
  return THCNumerics<BFloat16>::expm1(a);
}

inline __host__ __device__ BFloat16 pow(BFloat16 a, BFloat16 b) {
  return THCNumerics<BFloat16>::pow(a, b);
}

inline __host__ __device__ BFloat16 sqrt(BFloat16 a) {
  return THCNumerics<BFloat16>::sqrt(a);
}

inline __host__ __device__ BFloat16 tanh(BFloat16 a) {
  return THCNumerics<BFloat16>::tanh(a);
}

#endif
