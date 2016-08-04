#ifndef THC_NUMERICS_INC
#define THC_NUMERICS_INC

#include <cuda.h>
#include <limits.h>
#include "THCHalf.h"

/// Class for numeric limits of the particular data type, which
/// includes support for `half`.
/// Unfortunately since `half` does not have a constructor, these have
/// to be expressed as functions (either that or non-const statics).
template <typename T>
struct THCNumerics {
};

template <>
struct THCNumerics<unsigned char> {
  static inline __host__ __device__ unsigned char min() { return 0; }
  static inline __host__ __device__ unsigned char max() { return UCHAR_MAX; }

  static inline __host__ __device__ bool lt(unsigned char a, unsigned char b) { return a < b; }
  static inline __host__ __device__ bool le(unsigned char a, unsigned char b) { return a <= b; }
  static inline __host__ __device__ bool gt(unsigned char a, unsigned char b) { return a > b; }
  static inline __host__ __device__ bool ge(unsigned char a, unsigned char b) { return a >= b; }
  static inline __host__ __device__ bool eq(unsigned char a, unsigned char b) { return a == b; }
  static inline __host__ __device__ bool ne(unsigned char a, unsigned char b) { return a != b; }
};

template <>
struct THCNumerics<char> {
  static inline __host__ __device__ char min() { return CHAR_MIN; }
  static inline __host__ __device__ char max() { return CHAR_MAX; }

  static inline __host__ __device__ bool lt(char a, char b) { return a < b; }
  static inline __host__ __device__ bool le(char a, char b) { return a <= b; }
  static inline __host__ __device__ bool gt(char a, char b) { return a > b; }
  static inline __host__ __device__ bool ge(char a, char b) { return a >= b; }
  static inline __host__ __device__ bool eq(char a, char b) { return a == b; }
  static inline __host__ __device__ bool ne(char a, char b) { return a != b; }
};

template <>
struct THCNumerics<short> {
  static inline __host__ __device__ short min() { return SHRT_MIN; }
  static inline __host__ __device__ short max() { return SHRT_MAX; }

  static inline __host__ __device__ bool lt(short a, short b) { return a < b; }
  static inline __host__ __device__ bool le(short a, short b) { return a <= b; }
  static inline __host__ __device__ bool gt(short a, short b) { return a > b; }
  static inline __host__ __device__ bool ge(short a, short b) { return a >= b; }
  static inline __host__ __device__ bool eq(short a, short b) { return a == b; }
  static inline __host__ __device__ bool ne(short a, short b) { return a != b; }
};

template <>
struct THCNumerics<int> {
  static inline __host__ __device__ int min() { return INT_MIN; }
  static inline __host__ __device__ int max() { return INT_MAX; }

  static inline __host__ __device__ bool lt(int a, int b) { return a < b; }
  static inline __host__ __device__ bool le(int a, int b) { return a <= b; }
  static inline __host__ __device__ bool gt(int a, int b) { return a > b; }
  static inline __host__ __device__ bool ge(int a, int b) { return a >= b; }
  static inline __host__ __device__ bool eq(int a, int b) { return a == b; }
  static inline __host__ __device__ bool ne(int a, int b) { return a != b; }
};

template <>
struct THCNumerics<long> {
  static inline __host__ __device__ long min() { return LONG_MIN; }
  static inline __host__ __device__ long max() { return LONG_MAX; }

  static inline __host__ __device__ bool lt(long a, long b) { return a < b; }
  static inline __host__ __device__ bool le(long a, long b) { return a <= b; }
  static inline __host__ __device__ bool gt(long a, long b) { return a > b; }
  static inline __host__ __device__ bool ge(long a, long b) { return a >= b; }
  static inline __host__ __device__ bool eq(long a, long b) { return a == b; }
  static inline __host__ __device__ bool ne(long a, long b) { return a != b; }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct THCNumerics<half> {
  static inline __host__ __device__ half min() { half h; h.x = 0xfbff; return h; }
  static inline __host__ __device__ half max() { half h; h.x = 0x7bff; return h; }

  static inline __host__ __device__ bool lt(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hlt(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return fa < fb;
#endif
#else // __CUDA_ARCH__
    return THC_half2float(a) < THC_half2float(b);
#endif
  }

  static inline __host__ __device__ bool le(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hle(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return fa <= fb;
#endif
#else // __CUDA_ARCH__
    return THC_half2float(a) <= THC_half2float(b);
#endif
  }

  static inline __host__ __device__ bool gt(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hgt(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return fa > fb;
#endif
#else // __CUDA_ARCH__
    return THC_half2float(a) > THC_half2float(b);
#endif
  }

  static inline __host__ __device__ bool ge(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hge(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return fa >= fb;
#endif
#else // __CUDA_ARCH__
    return THC_half2float(a) >= THC_half2float(b);
#endif
  }

  static inline __host__ __device__ bool eq(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __heq(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return fa == fb;
#endif
#else // __CUDA_ARCH__
    return THC_half2float(a) == THC_half2float(b);
#endif
  }

  static inline __host__ __device__ bool ne(half a, half b) {
#ifdef __CUDA_ARCH__
#ifdef CUDA_HALF_INSTRUCTIONS
    return __hne(a, b);
#else
    float fa = __half2float(a);
    float fb = __half2float(b);
    return fa != fb;
#endif
#else // __CUDA_ARCH__
    return THC_half2float(a) != THC_half2float(b);
#endif
  }
};
#endif

template <>
struct THCNumerics<float> {
  static inline __host__ __device__ float min() { return -FLT_MAX; }
  static inline __host__ __device__ float max() { return FLT_MAX; }

  static inline __host__ __device__ bool lt(float a, float b) { return a < b; }
  static inline __host__ __device__ bool le(float a, float b) { return a <= b; }
  static inline __host__ __device__ bool gt(float a, float b) { return a > b; }
  static inline __host__ __device__ bool ge(float a, float b) { return a >= b; }
  static inline __host__ __device__ bool eq(float a, float b) { return a == b; }
  static inline __host__ __device__ bool ne(float a, float b) { return a != b; }
};

template <>
struct THCNumerics<double> {
  static inline __host__ __device__ double min() { return -DBL_MAX; }
  static inline __host__ __device__ double max() { return DBL_MAX; }

  static inline __host__ __device__ bool lt(double a, double b) { return a < b; }
  static inline __host__ __device__ bool le(double a, double b) { return a <= b; }
  static inline __host__ __device__ bool gt(double a, double b) { return a > b; }
  static inline __host__ __device__ bool ge(double a, double b) { return a >= b; }
  static inline __host__ __device__ bool eq(double a, double b) { return a == b; }
  static inline __host__ __device__ bool ne(double a, double b) { return a != b; }
};

/// `half` has some type conversion issues associated with it, since it
/// is a struct without a constructor/implicit conversion constructor.
/// We use this to convert scalar values to the given type that the
/// tensor expects.
template <typename In, typename Out>
struct ScalarConvert {
  static __host__ __device__ Out to(const In v) { return (Out) v; }
};

#ifdef CUDA_HALF_TENSOR
template <typename Out>
struct ScalarConvert<half, Out> {
  static __host__ __device__ Out to(const half v) {
#ifdef __CUDA_ARCH__
    return (Out) __half2float(v);
#else
    return (Out) THC_half2float(v);
#endif
  }
};

template <typename In>
struct ScalarConvert<In, half> {
  static __host__ __device__ half to(const In v) {
#ifdef __CUDA_ARCH__
    return __float2half((float) v);
#else
    return THC_float2half((float) v);
#endif
  }
};

template <>
struct ScalarConvert<half, half> {
  static __host__ __device__ half to(const half v) {
    return v;
  }
};
#endif

#endif // THC_TENSOR_TYPE_UTILS_INC
