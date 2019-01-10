#ifndef THC_HALF_AUTO_NUMERICS_INC
#define THC_HALF_AUTO_NUMERICS_INC

#include "THCHalf.h"
#include "THCNumerics.cuh"

// Half numerics functions defined as free functions, so cunn code can be
//written generically, i.e. without excessive calling of THCNumerics<half> functions.

// these functions should move to THCNumerics

#ifdef CUDA_HALF_TENSOR
inline __host__ __device__ half fmaxType(half x, half y) {
  return THCNumerics<half>::ge(x, y) ? x : y;
}

inline __host__ __device__ float fmaxType(float x, half y) {
  return fmaxf(x, ScalarConvert<half, float>::to(y));
}
#endif

inline __host__ __device__ float fmaxType(float x, float y) {
  return fmaxf(x, y);
}

inline __host__ __device__ double fmaxType(double x, double y) {
  return fmax(x, y);
}

#ifdef CUDA_HALF_TENSOR

// arithmetic functions

inline __host__ __device__ half operator+(half a, half b) {
  return THCNumerics<half>::add(a, b);
}

inline __host__ __device__ float operator+(half a, float b) {
  return ScalarConvert<half, float>::to(a) + b;
}

inline __host__ __device__ float operator+(float a, half b) {
  return a + ScalarConvert<half, float>::to(b);
}

inline __host__ __device__ double operator+(double a, half b) {
  return a + ScalarConvert<half, double>::to(b);
}

inline __host__ __device__ half operator-(half a) {
  return THCNumerics<half>::neg(a);
}

inline __host__ __device__ half operator-(half a, half b) {
  return THCNumerics<half>::add(a, THCNumerics<half>::neg(b));
}

inline __host__ __device__ half operator-(half a, int b) {
  return THCNumerics<half>::add(a, THCNumerics<half>::neg(ScalarConvert<int, half>::to(b)));
}

inline __host__ __device__ float operator-(half a, float b) {
  return ScalarConvert<half, float>::to(a) - b;
}

inline __host__ __device__ double operator-(half a, double b) {
  return ScalarConvert<half, double>::to(a) - b;
}

inline __host__ __device__ half operator-(int a, half b) {
  return THCNumerics<half>::add(ScalarConvert<int, half>::to(a), THCNumerics<half>::neg(b));
}

inline __host__ __device__ float operator-(float a, half b) {
  return a - ScalarConvert<half, float>::to(b);
}

inline __host__ __device__ double operator-(double a, half b) {
  return a - ScalarConvert<half, double>::to(b);
}

inline __host__ __device__ half operator*(half a, half b) {
  return THCNumerics<half>::mul(a, b);
}

inline __host__ __device__ float operator*(half a, float b) {
  return ScalarConvert<half, float>::to(a) * b;
}

inline __host__ __device__ double operator*(half a, double b) {
  return ScalarConvert<half, double>::to(a) * b;
}

inline __host__ __device__ half operator*(half a, int b) {
  return a * ScalarConvert<int, half>::to(b);
}

inline __host__ __device__ float operator*(float a, half b) {
  return a * ScalarConvert<half, float>::to(b);
}

inline __host__ __device__ double operator*(double a, half b) {
  return a * ScalarConvert<half, double>::to(b);
}

inline __host__ __device__ half operator/(half a, half b) {
  return THCNumerics<half>::div(a, b);
}

inline __host__ __device__ float operator/(float a, half b) {
  return a / ScalarConvert<half, float>::to(b);
}

inline __host__ __device__ double operator/(double a, half b) {
  return a / ScalarConvert<half, double>::to(b);
}

inline __host__ __device__ half operator/(int a, half b) {
  return ScalarConvert<int, half>::to(a) / b;
}

inline __host__ __device__ float operator/(half a, float b) {
  return ScalarConvert<half, float>::to(a) / b;
}

inline __host__ __device__ double operator/(half a, double b) {
  return ScalarConvert<half, double>::to(a) / b;
}

inline __host__ __device__ half operator/(half a, int b) {
  return a / ScalarConvert<int, half>::to(b);
}

inline __host__ __device__ half& operator+=(half &lhs, const half &rhs) {
  lhs = lhs + rhs;
  return lhs;
}
inline __host__ __device__ float& operator+=(float &lhs, const half &rhs) {
  lhs = lhs + rhs;
  return lhs;
}

inline __host__ __device__ float& operator-=(float &lhs, const half &rhs) {
  lhs = lhs - rhs;
  return lhs;
}

inline __host__ __device__ half& operator*=(half &lhs, const half &rhs) {
  lhs = lhs * rhs;
  return lhs;
}

inline __host__ __device__ half& operator/=(half &lhs, const int &rhs) {
  lhs = lhs / rhs;
  return lhs;
}

inline __host__ __device__ half& operator/=(half &lhs, const half &rhs) {
  lhs = lhs / rhs;
  return lhs;
}

inline __host__ __device__ half abs(half a) {
  return THCNumerics<half>::abs(a);
}

inline __host__ __device__ half exp(half a) {
  return THCNumerics<half>::exp(a);
}

inline __host__ __device__ half log10(half a) {
  return THCNumerics<half>::log10(a);
}

inline __host__ __device__ half log1p(half a) {
  return THCNumerics<half>::log1p(a);
}

inline __host__ __device__ half log2(half a) {
  return THCNumerics<half>::log2(a);
}

inline __host__ __device__ half expm1(half a) {
  return THCNumerics<half>::expm1(a);
}

inline __host__ __device__ half pow(half a, half b) {
  return THCNumerics<half>::pow(a, b);
}

inline __host__ __device__ half sqrt(half a) {
  return THCNumerics<half>::sqrt(a);
}

inline __host__ __device__ half tanh(half a) {
  return THCNumerics<half>::tanh(a);
}

#if defined(_MSC_VER) && CUDA_VERSION >= 9000
inline __host__ __device__ half operator+(half a, int b) {
  return THCNumerics<half>::add(a, ScalarConvert<int, half>::to(b));
}

inline __host__ __device__ double operator+(half a, double b) {
  return ScalarConvert<half, double>::to(a) + b;
}

inline __host__ __device__ half operator*(half a, bool b) {
  return THCNumerics<half>::mul(a, ScalarConvert<bool, half>::to(b));
}
#endif

// comparison functions

inline __host__ __device__ bool operator<(half a, half b) {
  return THCNumerics<half>::lt(a, b);
}

inline __host__ __device__ bool operator<=(half a, half b) {
  return THCNumerics<half>::le(a, b);
}

inline __host__ __device__ bool operator<=(half a, int b) {
  return THCNumerics<half>::le(a, ScalarConvert<int, half>::to(b));
}

inline __host__ __device__ bool operator<(half a, int b) {
  return THCNumerics<half>::lt(a, ScalarConvert<int, half>::to(b));
}

inline __host__ __device__ bool operator>(half a, half b) {
  return THCNumerics<half>::gt(a, b);
}

inline __host__ __device__ bool operator>(half a, int b) {
  return THCNumerics<half>::gt(a, ScalarConvert<int, half>::to(b));
}

inline __host__ __device__ bool operator>=(half a, half b) {
  return THCNumerics<half>::ge(a, b);
}

inline __host__ __device__ bool operator>=(half a, int b) {
  return THCNumerics<half>::ge(a, ScalarConvert<int ,half>::to(b));
}

#endif
#endif
