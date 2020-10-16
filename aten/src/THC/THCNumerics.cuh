#ifndef THC_NUMERICS_INC
#define THC_NUMERICS_INC

#include <cstdlib>
#include <limits>
#include <cuda.h>
#include <assert.h>
#include <TH/THHalf.h>
#include <ATen/ATen.h>
#include <ATen/cuda/NumericLimits.cuh>

// WARNING: THCNumerics is being deprecated. Please follow the comments
// in this file to learn about new usages.
// Comments on usage:
//      - lt,le,gt,ge,eq,neg,add,mul,sub,div and other binary ops can
//        be implemented using CUDA_apply_utils or binary cuda kernel
//      - Check NumericLimits.cuh for specialized math functions.
//      - Note how __half and at::Half can be casted. for instance:
//        static_cast<at::Half>(std::sin(static_cast<at::Half>(a)));

template <typename T>
struct THCNumerics {
};

template <typename T>
static inline __host__ __device__ T powi(T a, T b) {
  assert(THCNumerics<T>::ge(b, 0));
  T result = 1;
  while (b) {
    if (b & 1) {
       result *= a;
    }
    b /= 2;
    a *= a;
  }
  return result;
}

// DEPRECATED: For integral types, use math functions from std and NumericLimits.cuh.
//             Use binary_kernel or CUDA_apply_utils for arithmetic
template <>
struct THCNumerics<uint8_t> {
  static inline __host__ __device__ uint8_t min() { return at::numeric_limits<uint8_t>::lowest(); }
  static inline __host__ __device__ uint8_t max() { return at::numeric_limits<uint8_t>::max(); }
  static inline __host__ __device__ uint8_t lower_bound() { return at::numeric_limits<uint8_t>::lower_bound(); }
  static inline __host__ __device__ uint8_t upper_bound() { return at::numeric_limits<uint8_t>::upper_bound(); }

  static inline __host__ __device__ bool lt(uint8_t a, uint8_t b) { return a < b; }
  static inline __host__ __device__ bool le(uint8_t a, uint8_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(uint8_t a, uint8_t b) { return a > b; }
  static inline __host__ __device__ bool ge(uint8_t a, uint8_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(uint8_t a, uint8_t b) { return a == b; }
  static inline __host__ __device__ bool ne(uint8_t a, uint8_t b) { return a != b; }

  static inline __host__ __device__  uint8_t add(uint8_t a, uint8_t b) { return a + b; }
  static inline __host__ __device__  uint8_t mul(uint8_t a, uint8_t b) { return a * b; }
  static inline __host__ __device__  uint8_t sub(uint8_t a, uint8_t b) { return a - b; }
  static inline __host__ __device__  uint8_t div(uint8_t a, uint8_t b) { return a / b; }
  static inline __host__ __device__  uint8_t pow(uint8_t a, uint8_t b) { return powi<uint8_t>(a, b); }
  static inline __host__ __device__  bool isnan(uint8_t a) { return false; }
  static inline __host__ __device__  bool isinf(uint8_t a) { return false; }
};

#ifdef _MSC_VER
// Suppress warning C4804: '/': unsafe use of type 'bool' in operation
#pragma warning( push )
#pragma warning( disable : 4804 )
#endif

template <>
struct THCNumerics<bool> {
  static inline __host__ __device__ bool min() { return at::numeric_limits<bool>::lowest(); }
  static inline __host__ __device__ bool max() { return at::numeric_limits<bool>::max(); }
  static inline __host__ __device__ bool lower_bound() { return at::numeric_limits<bool>::lower_bound(); }
  static inline __host__ __device__ bool upper_bound() { return at::numeric_limits<bool>::upper_bound(); }

  static inline __host__ __device__ bool lt(bool a, bool b) { return a < b; }
  static inline __host__ __device__ bool le(bool a, bool b) { return a <= b; }
  static inline __host__ __device__ bool gt(bool a, bool b) { return a > b; }
  static inline __host__ __device__ bool ge(bool a, bool b) { return a >= b; }
  static inline __host__ __device__ bool eq(bool a, bool b) { return a == b; }
  static inline __host__ __device__ bool ne(bool a, bool b) { return a != b; }
  static inline __host__ __device__ bool add(bool a, bool b) { return a + b; }
  static inline __host__ __device__ bool mul(bool a, bool b) { return a && b; }
  static inline __host__ __device__ bool sub(bool a, bool b) { return a - b; }
  static inline __host__ __device__ bool div(bool a, bool b) { return a / b; }
  static inline __host__ __device__ bool isnan(bool a) { return false; }
  static inline __host__ __device__ bool isinf(bool a) { return false; }
};

#ifdef _MSC_VER
#pragma warning( pop )
#endif

template <>
struct THCNumerics<int8_t> {
  static inline __host__ __device__ int8_t min() { return at::numeric_limits<int8_t>::lowest(); }
  static inline __host__ __device__ int8_t max() { return at::numeric_limits<int8_t>::max(); }
  static inline __host__ __device__ int8_t lower_bound() { return at::numeric_limits<int8_t>::lower_bound(); }
  static inline __host__ __device__ int8_t upper_bound() { return at::numeric_limits<int8_t>::upper_bound(); }

  static inline __host__ __device__ bool lt(int8_t a, int8_t b) { return a < b; }
  static inline __host__ __device__ bool le(int8_t a, int8_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(int8_t a, int8_t b) { return a > b; }
  static inline __host__ __device__ bool ge(int8_t a, int8_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(int8_t a, int8_t b) { return a == b; }
  static inline __host__ __device__ bool ne(int8_t a, int8_t b) { return a != b; }

  static inline __host__ __device__  int8_t add(int8_t a, int8_t b) { return a + b; }
  static inline __host__ __device__  int8_t mul(int8_t a, int8_t b) { return a * b; }
  static inline __host__ __device__  int8_t sub(int8_t a, int8_t b) { return a - b; }
  static inline __host__ __device__  int8_t div(int8_t a, int8_t b) { return a / b; }
  static inline __host__ __device__  int8_t pow(int8_t a, int8_t b) { return powi<int8_t>(a, b); }
  static inline __host__ __device__  bool isnan(int8_t a) { return false; }
  static inline __host__ __device__  bool isinf(int8_t a) { return false; }
};

template <>
struct THCNumerics<int16_t> {
  static inline __host__ __device__ int16_t min() { return at::numeric_limits<int16_t>::lowest(); }
  static inline __host__ __device__ int16_t max() { return at::numeric_limits<int16_t>::max(); }
  static inline __host__ __device__ int16_t lower_bound() { return at::numeric_limits<int16_t>::lower_bound(); }
  static inline __host__ __device__ int16_t upper_bound() { return at::numeric_limits<int16_t>::upper_bound(); }

  static inline __host__ __device__ bool lt(int16_t a, int16_t b) { return a < b; }
  static inline __host__ __device__ bool le(int16_t a, int16_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(int16_t a, int16_t b) { return a > b; }
  static inline __host__ __device__ bool ge(int16_t a, int16_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(int16_t a, int16_t b) { return a == b; }
  static inline __host__ __device__ bool ne(int16_t a, int16_t b) { return a != b; }

  static inline __host__ __device__  int16_t add(int16_t a, int16_t b) { return a + b; }
  static inline __host__ __device__  int16_t mul(int16_t a, int16_t b) { return a * b; }
  static inline __host__ __device__  int16_t sub(int16_t a, int16_t b) { return a - b; }
  static inline __host__ __device__  int16_t div(int16_t a, int16_t b) { return a / b; }
  static inline __host__ __device__  int16_t pow(int16_t a, int16_t b) { return powi<int16_t>(a, b); }
  static inline __host__ __device__  bool isnan(int16_t a) { return false; }
  static inline __host__ __device__  bool isinf(int16_t a) { return false; }
};

template <>
struct THCNumerics<int32_t> {
  static inline __host__ __device__ int32_t min() { return at::numeric_limits<int32_t>::lowest(); }
  static inline __host__ __device__ int32_t max() { return at::numeric_limits<int32_t>::max(); }
  static inline __host__ __device__ int32_t lower_bound() { return at::numeric_limits<int32_t>::lower_bound(); }
  static inline __host__ __device__ int32_t upper_bound() { return at::numeric_limits<int32_t>::upper_bound(); }

  static inline __host__ __device__ bool lt(int32_t a, int32_t b) { return a < b; }
  static inline __host__ __device__ bool le(int32_t a, int32_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(int32_t a, int32_t b) { return a > b; }
  static inline __host__ __device__ bool ge(int32_t a, int32_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(int32_t a, int32_t b) { return a == b; }
  static inline __host__ __device__ bool ne(int32_t a, int32_t b) { return a != b; }

  static inline __host__ __device__  int32_t add(int32_t a, int32_t b) { return a + b; }
  static inline __host__ __device__  int32_t mul(int32_t a, int32_t b) { return a * b; }
  static inline __host__ __device__  int32_t sub(int32_t a, int32_t b) { return a - b; }
  static inline __host__ __device__  int32_t div(int32_t a, int32_t b) { return a / b; }
  static inline __host__ __device__  int32_t pow(int32_t a, int32_t b) { return powi<int32_t>(a, b); }
  static inline __host__ __device__  bool isnan(int32_t a) { return false; }
  static inline __host__ __device__  bool isinf(int32_t a) { return false; }
};

template <>
struct THCNumerics<int64_t> {
  static inline __host__ __device__ int64_t min() { return at::numeric_limits<int64_t>::lowest(); }
  static inline __host__ __device__ int64_t max() { return at::numeric_limits<int64_t>::max(); }
  static inline __host__ __device__ int64_t lower_bound() { return at::numeric_limits<int64_t>::lower_bound(); }
  static inline __host__ __device__ int64_t upper_bound() { return at::numeric_limits<int64_t>::upper_bound(); }

  static inline __host__ __device__ bool lt(int64_t a, int64_t b) { return a < b; }
  static inline __host__ __device__ bool le(int64_t a, int64_t b) { return a <= b; }
  static inline __host__ __device__ bool gt(int64_t a, int64_t b) { return a > b; }
  static inline __host__ __device__ bool ge(int64_t a, int64_t b) { return a >= b; }
  static inline __host__ __device__ bool eq(int64_t a, int64_t b) { return a == b; }
  static inline __host__ __device__ bool ne(int64_t a, int64_t b) { return a != b; }


  static inline __host__ __device__  int64_t add(int64_t a, int64_t b) { return a + b; }
  static inline __host__ __device__  int64_t mul(int64_t a, int64_t b) { return a * b; }
  static inline __host__ __device__  int64_t sub(int64_t a, int64_t b) { return a - b; }
  static inline __host__ __device__  int64_t div(int64_t a, int64_t b) { return a / b; };
  static inline __host__ __device__  int64_t pow(int64_t a, int64_t b) { return powi<int64_t>(a, b); }
  static inline __host__ __device__  bool isnan(int64_t a) { return false; }
  static inline __host__ __device__  bool isinf(int64_t a) { return false; }
};

// DEPRECATED: use math functions from std and NumericLimits.cuh
template <>
struct THCNumerics<at::Half> {
  static inline __host__ __device__ at::Half min() { return at::numeric_limits<at::Half>::lowest(); }
  static inline __host__ __device__ at::Half max() { return at::numeric_limits<at::Half>::max(); }
  static inline __host__ __device__ at::Half lower_bound() { return at::numeric_limits<at::Half>::lower_bound(); }
  static inline __host__ __device__ at::Half upper_bound() { return at::numeric_limits<at::Half>::upper_bound(); }

  static inline __host__ __device__ bool lt(at::Half a, at::Half b) { return a < b; }
  static inline __host__ __device__ bool le(at::Half a, at::Half b) { return a <= b; }
  static inline __host__ __device__ bool gt(at::Half a, at::Half b) { return a > b; }
  static inline __host__ __device__ bool ge(at::Half a, at::Half b) { return a >= b; }
  static inline __host__ __device__ bool eq(at::Half a, at::Half b) { return a == b; }
  static inline __host__ __device__ bool ne(at::Half a, at::Half b) { return a != b; }

  static inline __host__ __device__ at::Half sqrt(at::Half a) { return ::sqrt(a); }
  static inline __host__ __device__ at::Half atan(at::Half a) { return ::atan(a); }
  static inline __host__ __device__ at::Half add(at::Half a, at::Half b) { return a + b; }
  static inline __host__ __device__ at::Half div(at::Half a, at::Half b) { return a / b; }
  static inline __host__ __device__ at::Half mul(at::Half a, at::Half b) { return a * b; }
  static inline __host__ __device__ at::Half sub(at::Half a, at::Half b) { return a - b; }
  static inline __host__ __device__ at::Half pow(at::Half a, at::Half b) { return ::pow(a, b); }

  static inline __host__ __device__ bool isnan(at::Half a) {
    #ifdef _MSC_VER
      // Windows requires this explicit conversion. The reason is unclear
      // related issue with clang: https://reviews.llvm.org/D37906
      return ::isnan((float) a);
    #else
      return ::isnan(a);
    #endif
  }

  static inline __host__ __device__ bool isinf(at::Half a) {
    #ifdef _MSC_VER
      // Windows requires this explicit conversion. The reason is unclear
      // related issue with clang: https://reviews.llvm.org/D37906
      return ::isinf((float) a);
    #else
      return ::isinf(a);
    #endif
  }

};

// DEPRECATED: use math functions from std and cuda math API (if needed)
template <>
struct THCNumerics<float> {
  static inline __host__ __device__ float min() { return at::numeric_limits<float>::lowest(); }
  static inline __host__ __device__ float max() { return at::numeric_limits<float>::max(); }
  static inline __host__ __device__ float lower_bound() { return at::numeric_limits<float>::lower_bound(); }
  static inline __host__ __device__ float upper_bound() { return at::numeric_limits<float>::upper_bound(); }

  static inline __host__ __device__ bool lt(float a, float b) { return a < b; }
  static inline __host__ __device__ bool le(float a, float b) { return a <= b; }
  static inline __host__ __device__ bool gt(float a, float b) { return a > b; }
  static inline __host__ __device__ bool ge(float a, float b) { return a >= b; }
  static inline __host__ __device__ bool eq(float a, float b) { return a == b; }
  static inline __host__ __device__ bool ne(float a, float b) { return a != b; }

  static inline __host__ __device__  float sqrt (float a) { return  sqrtf(a); }
  static inline __host__ __device__  float atan (float a) { return  atanf(a); }
  static inline __host__ __device__  float add  (float a, float b) { return a + b; }
  static inline __host__ __device__  float div  (float a, float b) { return a / b; }
  static inline __host__ __device__  float mul  (float a, float b) { return a * b; }
  static inline __host__ __device__  float sub  (float a, float b) { return a - b; }
  static inline __host__ __device__  float pow  (float a, float b) { return powf(a, b); }
  static inline __host__ __device__  bool isnan(float a) { return ::isnan(a); }
  static inline __host__ __device__  bool isinf(float a) { return ::isinf(a); }
};

template <>
struct THCNumerics<at::BFloat16> {
  static inline __host__ __device__ at::BFloat16 min() { return at::numeric_limits<at::BFloat16>::lowest(); }
  static inline __host__ __device__ at::BFloat16 max() { return at::numeric_limits<at::BFloat16>::max(); }
  static inline __host__ __device__ at::BFloat16 lower_bound() { return at::numeric_limits<at::BFloat16>::lower_bound(); }
  static inline __host__ __device__ at::BFloat16 upper_bound() { return at::numeric_limits<at::BFloat16>::upper_bound(); }

  static inline __host__ __device__ bool lt(at::BFloat16 a, at::BFloat16 b) { return a < b; }
  static inline __host__ __device__ bool le(at::BFloat16 a, at::BFloat16 b) { return a <= b; }
  static inline __host__ __device__ bool gt(at::BFloat16 a, at::BFloat16 b) { return a > b; }
  static inline __host__ __device__ bool ge(at::BFloat16 a, at::BFloat16 b) { return a >= b; }
  static inline __host__ __device__ bool eq(at::BFloat16 a, at::BFloat16 b) { return a == b; }
  static inline __host__ __device__ bool ne(at::BFloat16 a, at::BFloat16 b) { return a != b; }

  static inline __host__ __device__  at::BFloat16 sqrt (at::BFloat16 a) { return  sqrtf(a); }
  static inline __host__ __device__  at::BFloat16 atan (at::BFloat16 a) { return  atanf(a); }
  static inline __host__ __device__  at::BFloat16 add  (at::BFloat16 a, at::BFloat16 b) { return a + b; }
  static inline __host__ __device__  at::BFloat16 div  (at::BFloat16 a, at::BFloat16 b) { return a / b; }
  static inline __host__ __device__  at::BFloat16 mul  (at::BFloat16 a, at::BFloat16 b) { return a * b; }
  static inline __host__ __device__  at::BFloat16 sub  (at::BFloat16 a, at::BFloat16 b) { return a - b; }
  static inline __host__ __device__  at::BFloat16 pow  (at::BFloat16 a, at::BFloat16 b) { return powf(a, b); }
  static inline __host__ __device__  at::BFloat16 atan2(at::BFloat16 a, at::BFloat16 b) { return atan2f(a, b); }

  static inline __host__ __device__ bool isnan(at::BFloat16 a) {
    #ifdef _MSC_VER
      // Windows requires this explicit conversion. The reason is unclear
      // related issue with clang: https://reviews.llvm.org/D37906
      return ::isnan((float) a);
    #else
      return ::isnan(a);
    #endif
  }

  static inline __host__ __device__ bool isinf(at::BFloat16 a) {
    #ifdef _MSC_VER
      // Windows requires this explicit conversion. The reason is unclear
      // related issue with clang: https://reviews.llvm.org/D37906
      return ::isinf((float) a);
    #else
      return ::isinf(a);
    #endif
  }
};

// DEPRECATED: use math functions from std and cuda math API (if needed)
template <>
struct THCNumerics<double> {
  static inline __host__ __device__ double min() { return at::numeric_limits<double>::lowest(); }
  static inline __host__ __device__ double max() { return at::numeric_limits<double>::max(); }
  static inline __host__ __device__ double lower_bound() { return at::numeric_limits<double>::lower_bound(); }
  static inline __host__ __device__ double upper_bound() { return at::numeric_limits<double>::upper_bound(); }

  static inline __host__ __device__ bool lt(double a, double b) { return a < b; }
  static inline __host__ __device__ bool le(double a, double b) { return a <= b; }
  static inline __host__ __device__ bool gt(double a, double b) { return a > b; }
  static inline __host__ __device__ bool ge(double a, double b) { return a >= b; }
  static inline __host__ __device__ bool eq(double a, double b) { return a == b; }
  static inline __host__ __device__ bool ne(double a, double b) { return a != b; }

  static inline __host__ __device__  double sqrt (double a) { return  ::sqrt(a); }
  static inline __host__ __device__  double atan (double a) { return  ::atan(a); }
  static inline __host__ __device__  double add  (double a, double b) { return a + b; }
  static inline __host__ __device__  double div  (double a, double b) { return a / b; }
  static inline __host__ __device__  double mul  (double a, double b) { return a * b; }
  static inline __host__ __device__  double sub  (double a, double b) { return a - b; }
  static inline __host__ __device__  double pow  (double a, double b) { return ::pow(a, b); }
  static inline __host__ __device__  bool isnan(double a) { return ::isnan(a); }
  static inline __host__ __device__  bool isinf(double a) { return ::isinf(a); }
};

// WARNING: The following note is deprecated
///       `half` has some type conversion issues associated with it, since it
///        is a struct without a constructor/implicit conversion constructor.
///        We use this to convert scalar values to the given type that the
///        tensor expects.
///
/// at::Half has implicit conversions for float and __half types. Moreover
/// it has constructors for __half and float types.

template <typename In, typename Out>
struct ScalarConvert {
  static __host__ __device__ Out to(const In v) { return (Out) v; }
};

// DEPRECATED: use static_cast in kernels instead of scalar_cast
template <typename T, typename U>
__host__ __device__ T scalar_cast(U u) {
  return ScalarConvert<U, T>::to(u);
}

#endif // THC_NUMERICS_INC
