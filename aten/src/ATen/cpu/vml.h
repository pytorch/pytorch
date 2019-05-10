#pragma once

#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>

// This header implements various unary operations using a MKL VML style
// interface.

// It implements various functions with a simple interface
// For example it enables the user to call vsin(float* out, const float* in,
// size) This functions takes a pointer to a contious output array of floats and
// a constant input array. It will then apply sin to each value in in the input
// array and write the result into the output array. out and in may point to the
// same memory, i.e. this fully supports in-place operations. These functions
// also implement their own parallelization, so take precautions when calling
// these from threaded functions.

// When MKL is available it will call into MKL's VML library similar to NumPy
// If MKL is not available it will use SLEEF.

// This file might be compiled under AVX or AVX2 when called from e.g.
// UnaryOpsKernel.cpp

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>

#if AT_MKL_ENABLED() && !defined(__APPLE__)
#include <mkl.h>
#endif

// [Note SSE-AVX transitions]
// There is a bug in Glibc2.23
// https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/1663280. Calling zeroall
// when using AVX/AVX2 code resolves this.
#if defined(__AVX__) && defined(__GLIBC__) && __GLIBC_MINOR__ == 23
#define DL_RUNTIME_BUG(op, type) \
  volatile type x = (type)(1);   \
  x = std::op(x);                \
  _mm256_zeroall();
#else
#define DL_RUNTIME_BUG(op, type)
#endif

namespace at {
namespace vml {
namespace {

using namespace vec256;

template <typename scalar_t>
inline void vrsqrt(scalar_t* out, scalar_t* in, int64_t size) {
  parallel_for(0, size, 2048, [out, in](int64_t begin, int64_t end) {
    map(
        [](const Vec256<scalar_t>& x) {
          return Vec256<scalar_t>((scalar_t)(1)) / x.sqrt();
        },
        out + begin,
        in + begin,
        end - begin);
  });
}

// NB: We ignore numerical errors by convention and leave them to the user

// We unfortunately need to duplicate code here to deal with the SSE-AVX
// transition bug (see [Note SSE-AVX transitions]). As soon as we can expect
// users to use a version of glibc newer than 2.23 we will be able to ditch
// this. This duplication is also necessary since not all functions (e.g. rsqrt)
// might be part of cmath.

#define IMPLEMENT_VML_BUG(op)                                          \
  template <typename scalar_t>                                          \
  inline void v##op(scalar_t* out, const scalar_t* in, int64_t size) {  \
    DL_RUNTIME_BUG(op, scalar_t)                                        \
    parallel_for(0, size, 2048, [out, in](int64_t begin, int64_t end) { \
      map([](const Vec256<scalar_t>& x) { return x.op(); },             \
          out + begin,                                                  \
          in + begin,                                                   \
          end - begin);                                                 \
    });                                                                 \
  }

#define IMPLEMENT_VML(op)                                              \
  template <typename scalar_t>                                          \
  inline void v##op(scalar_t* out, const scalar_t* in, int64_t size) {  \
    parallel_for(0, size, 2048, [out, in](int64_t begin, int64_t end) { \
      map([](const Vec256<scalar_t>& x) { return x.op(); },             \
          out + begin,                                                  \
          in + begin,                                                   \
          end - begin);                                                 \
    });                                                                 \
  }

IMPLEMENT_VML_BUG(abs)
IMPLEMENT_VML_BUG(acos)
IMPLEMENT_VML_BUG(asin)
IMPLEMENT_VML_BUG(atan)
IMPLEMENT_VML_BUG(ceil)
IMPLEMENT_VML_BUG(cos)
// IMPLEMENT_VML_BUG(cosh)
IMPLEMENT_VML_BUG(erf)
IMPLEMENT_VML_BUG(erfc)
IMPLEMENT_VML_BUG(exp)
IMPLEMENT_VML_BUG(expm1)
IMPLEMENT_VML_BUG(floor)
IMPLEMENT_VML(reciprocal)
IMPLEMENT_VML_BUG(log)
IMPLEMENT_VML_BUG(log10)
IMPLEMENT_VML_BUG(log1p)
IMPLEMENT_VML_BUG(log2)
IMPLEMENT_VML(neg)
IMPLEMENT_VML_BUG(sin)
// IMPLEMENT_VML_BUG(sinh)
IMPLEMENT_VML_BUG(sqrt)
IMPLEMENT_VML_BUG(round)
IMPLEMENT_VML(rsqrt)
IMPLEMENT_VML_BUG(tan)
IMPLEMENT_VML_BUG(tanh)
IMPLEMENT_VML_BUG(trunc)

#if AT_MKL_ENABLED() && !defined(__APPLE__)

// NB: LP64 MKL is the most commonly used and thus we assume it here. That means
// we need to expect MKL_INT to be of type int, which implies int32_t in most
// cases.
static_assert(
    std::is_same<MKL_INT, int32_t>::value,
    "MKL_INT is assumed to be int32_t");
#define IMPLEMENT_VML_MKL_STUB(op, mklop, type, mkltype)                    \
  template <>                                                           \
  inline void v##op(type * out, const type * in, int64_t size) {          \
    int64_t max_mkl_ind = std::numeric_limits<MKL_INT>::max();          \
    if (size <= static_cast<int64_t>(max_mkl_ind)) {                    \
      vm##mkltype##mklop(                                               \
          size, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE); \
    } else {                                                            \
      MKL_INT ind = 0;                                                  \
      int64_t chunks = size / max_mkl_ind;                              \
      int64_t rest = size % max_mkl_ind;                                \
      for (; ind < chunks; ind++) {                                     \
        vm##mkltype##mklop(                                             \
            max_mkl_ind,                                                \
            in + ind * max_mkl_ind,                                     \
            out + ind * max_mkl_ind,                                    \
            VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);              \
      }                                                                 \
      vm##mkltype##mklop(                                               \
          rest,                                                         \
          in + ind * max_mkl_ind,                                       \
          out + ind * max_mkl_ind,                                      \
          VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);                \
    }                                                                   \
  }

#define IMPLEMENT_VML_MKL(op, mklop)          \
  IMPLEMENT_VML_MKL_STUB(op, mklop, float, s) \
  IMPLEMENT_VML_MKL_STUB(op, mklop, double, d)

// NB: abs, cosh and sinh were temporarily disabled due to issues with Apple clang

IMPLEMENT_VML_MKL(abs, Abs)
IMPLEMENT_VML_MKL(acos, Acos)
IMPLEMENT_VML_MKL(asin, Asin)
IMPLEMENT_VML_MKL(atan, Atan)
IMPLEMENT_VML_MKL(cos, Cos)
// IMPLEMENT_VML_MKL(cosh, Cosh)
IMPLEMENT_VML_MKL(erf, Erf)
IMPLEMENT_VML_MKL(erfc, Erfc)
IMPLEMENT_VML_MKL(exp, Exp)
IMPLEMENT_VML_MKL(expm1, Expm1)
IMPLEMENT_VML_MKL(log, Ln)
IMPLEMENT_VML_MKL(log10, Log10)
IMPLEMENT_VML_MKL(log1p, Log1p)
IMPLEMENT_VML_MKL(sin, Sin)
// IMPLEMENT_VML_MKL(sinh, Sinh)
IMPLEMENT_VML_MKL(sqrt, Sqrt)
IMPLEMENT_VML_MKL(tan, Tan)
IMPLEMENT_VML_MKL(tanh, Tanh)
IMPLEMENT_VML_MKL(trunc, Trunc)

#if INTEL_MKL_VERSION >= 20180406
IMPLEMENT_VML_MKL(log2, Log2)
#endif

#endif

} // namespace
} // namespace vml
} // namespace at
