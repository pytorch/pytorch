#ifndef THC_HALF_AUTO_NUMERICS_INC
#define THC_HALF_AUTO_NUMERICS_INC

#include "THCHalf.h"
#include "THCNumerics.cuh"

// WARNING: THCNumerics is being deprecated. Read the comments and function usage
//          in THCNumerics to learn about the deprecation
//
// Half numerics functions defined as free functions, so cunn code can be
// written generically, i.e. without excessive calling of THCNumerics<THCHalf> functions.

// these functions should move to THCNumerics

inline __host__ __device__ THCHalf fmaxType(THCHalf x, THCHalf y) {
  return THCNumerics<THCHalf>::ge(x, y) ? x : y;
}

inline __host__ __device__ float fmaxType(float x, THCHalf y) {
  return fmaxf(x, ScalarConvert<THCHalf, float>::to(y));
}

inline __host__ __device__ float fmaxType(float x, float y) {
  return fmaxf(x, y);
}

inline __host__ __device__ double fmaxType(double x, double y) {
  return fmax(x, y);
}


// arithmetic functions

inline __host__ __device__ THCHalf abs(THCHalf a) {
  return THCNumerics<THCHalf>::abs(a);
}

inline __host__ __device__ THCHalf exp(THCHalf a) {
  return THCNumerics<THCHalf>::exp(a);
}

inline __host__ __device__ THCHalf log10(THCHalf a) {
  return THCNumerics<THCHalf>::log10(a);
}

inline __host__ __device__ THCHalf log1p(THCHalf a) {
  return THCNumerics<THCHalf>::log1p(a);
}

inline __host__ __device__ THCHalf log2(THCHalf a) {
  return THCNumerics<THCHalf>::log2(a);
}

inline __host__ __device__ THCHalf expm1(THCHalf a) {
  return THCNumerics<THCHalf>::expm1(a);
}

inline __host__ __device__ THCHalf pow(THCHalf a, THCHalf b) {
  return THCNumerics<THCHalf>::pow(a, b);
}

inline __host__ __device__ THCHalf sqrt(THCHalf a) {
  return THCNumerics<THCHalf>::sqrt(a);
}

inline __host__ __device__ THCHalf tanh(THCHalf a) {
  return THCNumerics<THCHalf>::tanh(a);
}

#endif
