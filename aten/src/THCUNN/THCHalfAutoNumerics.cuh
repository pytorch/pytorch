#ifndef THC_HALF_AUTO_NUMERICS_INC
#define THC_HALF_AUTO_NUMERICS_INC

#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>

// WARNING: THCNumerics is being deprecated. Read the comments and function usage
//          in THCNumerics to learn about the deprecation
//
// Half numerics functions defined as free functions, so cunn code can be
// written generically, i.e. without excessive calling of THCNumerics<THHalf> functions.

// these functions should move to THCNumerics

inline __host__ __device__ THHalf fmaxType(THHalf x, THHalf y) {
  return THCNumerics<THHalf>::ge(x, y) ? x : y;
}

inline __host__ __device__ float fmaxType(float x, THHalf y) {
  return fmaxf(x, ScalarConvert<THHalf, float>::to(y));
}

inline __host__ __device__ float fmaxType(float x, float y) {
  return fmaxf(x, y);
}

inline __host__ __device__ double fmaxType(double x, double y) {
  return fmax(x, y);
}


// arithmetic functions

inline __host__ __device__ THHalf abs(THHalf a) {
  return THCNumerics<THHalf>::abs(a);
}

inline __host__ __device__ THHalf exp(THHalf a) {
  return THCNumerics<THHalf>::exp(a);
}

inline __host__ __device__ THHalf log10(THHalf a) {
  return THCNumerics<THHalf>::log10(a);
}

inline __host__ __device__ THHalf log1p(THHalf a) {
  return THCNumerics<THHalf>::log1p(a);
}

inline __host__ __device__ THHalf log2(THHalf a) {
  return THCNumerics<THHalf>::log2(a);
}

inline __host__ __device__ THHalf pow(THHalf a, THHalf b) {
  return THCNumerics<THHalf>::pow(a, b);
}

inline __host__ __device__ THHalf sqrt(THHalf a) {
  return THCNumerics<THHalf>::sqrt(a);
}

inline __host__ __device__ THHalf tanh(THHalf a) {
  return THCNumerics<THHalf>::tanh(a);
}

#endif
