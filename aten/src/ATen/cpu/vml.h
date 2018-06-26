#pragma once

#include "ATen/Config.h"
#include "ATen/Parallel.h"
#include "ATen/cpu/vec256/functional.h"
#include "ATen/cpu/vec256/vec256.h"

// This header implements various unary operations using a MKL VML style
// interface.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

#if AT_MKL_ENABLED() && !defined(__APPLE__)
#include <mkl.h>
#include <mkl_vml.h>
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

#define IMPLEMENT_VML(op)                                               \
  template <typename scalar_t>                                          \
  inline void v##op(scalar_t* out, scalar_t* in, int64_t size) {        \
    parallel_for(0, size, 2048, [out, in](int64_t begin, int64_t end) { \
      map([](const Vec256<scalar_t>& x) { return x.op(); },             \
          out + begin,                                                  \
          in + begin,                                                   \
          end - begin);                                                 \
    });                                                                 \
  }

#define IMPLEMENT_FLOAT_MKL_VML(op, mklop)                                    \
  template <typename scalar_t>                                                \
  inline void v##op(scalar_t* out, scalar_t* in, int64_t size);               \
  template <>                                                                 \
  inline void v##op(float* out, float* in, int64_t size) {                    \
    vms##mklop(size, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE); \
  }                                                                           \
  template <>                                                                 \
  inline void v##op(double* out, double* in, int64_t size) {                  \
    vmd##mklop(size, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE); \
  }

// NB: abs, cosh and sinh were temporarily disabled due to issues with Apple clang

#if AT_MKL_ENABLED() && !defined(__APPLE__)
IMPLEMENT_FLOAT_MKL_VML(acos, Acos)
IMPLEMENT_FLOAT_MKL_VML(asin, Asin)
IMPLEMENT_FLOAT_MKL_VML(atan, Atan)
IMPLEMENT_FLOAT_MKL_VML(cos, Cos)
// IMPLEMENT_FLOAT_MKL_VML(cosh, Cosh)
IMPLEMENT_FLOAT_MKL_VML(erf, Erf)
IMPLEMENT_FLOAT_MKL_VML(exp, Exp)
IMPLEMENT_FLOAT_MKL_VML(expm1, Expm1)
IMPLEMENT_FLOAT_MKL_VML(log, Ln)
IMPLEMENT_FLOAT_MKL_VML(log10, Log10)
IMPLEMENT_FLOAT_MKL_VML(log1p, Log1p)
IMPLEMENT_FLOAT_MKL_VML(sin, Sin)
// IMPLEMENT_FLOAT_MKL_VML(sinh, Sinh)
IMPLEMENT_FLOAT_MKL_VML(sqrt, Sqrt)
IMPLEMENT_FLOAT_MKL_VML(tan, Tan)
IMPLEMENT_FLOAT_MKL_VML(tanh, Tanh)
IMPLEMENT_FLOAT_MKL_VML(trunc, Trunc)

#if INTEL_MKL_VERSION >= 20180406
IMPLEMENT_FLOAT_MKL_VML(log2, Log2)
#else
IMPLEMENT_VML(log2)
#endif

#else
IMPLEMENT_VML(acos)
IMPLEMENT_VML(asin)
IMPLEMENT_VML(atan)
IMPLEMENT_VML(cos)
// IMPLEMENT_VML(cosh)
IMPLEMENT_VML(erf)
IMPLEMENT_VML(exp)
IMPLEMENT_VML(expm1)
IMPLEMENT_VML(log)
IMPLEMENT_VML(log10)
IMPLEMENT_VML(log1p)
IMPLEMENT_VML(log2)
IMPLEMENT_VML(sin)
// IMPLEMENT_VML(sinh)
IMPLEMENT_VML(sqrt)
IMPLEMENT_VML(tan)
IMPLEMENT_VML(tanh)
#endif

IMPLEMENT_VML(ceil)
IMPLEMENT_VML(floor)
IMPLEMENT_VML(round)
IMPLEMENT_VML(trunc)

} // namespace
} // namespace vml
} // namespace at
