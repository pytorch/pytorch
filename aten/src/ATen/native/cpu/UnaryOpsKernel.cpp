#include "ATen/native/cpu/UnaryOpsKernel.h"

#include <cmath>
#include "ATen/Dispatch.h"
#include "ATen/cpu/vml.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/native/DispatchStub.h"
#ifdef __AVX2__
#include "ATen/native/cpu/avx_mathfun.h"
#endif

namespace at { namespace native {
namespace {

using namespace vec256;

template <typename scalar_t>
static int64_t _sigmoid(scalar_t* x, scalar_t* y, int64_t size);

// This should be a temporary solution until we understand why SLEEF is slower
// for sigmoid

template <>
int64_t _sigmoid(float* x, float* y, int64_t size) {
  using Vec = Vec256<float>;
  int64_t i = 0;
  for (; i < size - (size % (2 * Vec::size)); i += 2 * Vec::size) {
    Vec ret = Vec::loadu(y + i);
    Vec ret2 = Vec::loadu(y + i + Vec::size);
    ret = ret.neg();
    ret2 = ret2.neg();
#if defined(__AVX2__) && !defined(_MSC_VER)
    ret = exp256_ps(ret);
    ret2 = exp256_ps(ret2);
#else
    ret = ret.exp();
    ret2 = ret2.exp();
#endif
    ret = Vec((float)(1)) + ret;
    ret2 = Vec((float)(1)) + ret2;
    ret = ret.reciprocal();
    ret2 = ret2.reciprocal();
    ret.store(x + i);
    ret2.store(x + i + Vec::size);
  }
  return i;
}

template <>
int64_t _sigmoid(double* x, double* y, int64_t size) {
  using Vec = Vec256<double>;
  int64_t i = 0;
  for (; i < size - (size % (2 * Vec::size)); i += 2 * Vec::size) {
    Vec ret = Vec::loadu(y + i);
    Vec ret2 = Vec::loadu(y + i + Vec::size);
    ret = ret.neg();
    ret2 = ret2.neg();
    ret = ret.exp();
    ret2 = ret2.exp();
    ret = Vec((double)(1)) + ret;
    ret2 = Vec((double)(1)) + ret2;
    ret = ret.reciprocal();
    ret2 = ret2.reciprocal();
    ret.store(x + i);
    ret2.store(x + i + Vec::size);
  }
  return i;
}

static void sigmoid_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "sigmoid", [&] {
    using Vec = Vec256<scalar_t>;
    CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(
        result,
        self,
        [](int64_t size,
           scalar_t* x,
           scalar_t* y,
           int64_t stridex,
           int64_t stridey) {
          int64_t i = 0;
          if (stridex == 1 && stridey == 1) {
            i = _sigmoid(x, y, size);
          }
          for (; i < size; i += Vec::size) {
            scalar_t buffer[Vec::size];
            int64_t width = Vec::size;
            width = std::min(width, size - i);
            for (int64_t j = 0; j < width; j++) {
              buffer[j] = y[stridey * (i + j)];
            }
            Vec ret = Vec::loadu(buffer);
            ret = Vec((scalar_t)(0)) - ret;
            ret = ret.exp();
            ret = Vec((scalar_t)(1)) + ret;
            ret = ret.reciprocal();
            ret.store(buffer);
            for (int64_t j = 0; j < width; j++)
              x[stridex * (i + j)] = buffer[j];
          }
        });
  });
}

#define IMPLEMENT_FLOAT_KERNEL(dispatchtypes, op)                          \
  static void op##_kernel(Tensor& result, const Tensor& self) {            \
    checkBackend(#op, {result}, kCPU);                                     \
    AT_DISPATCH_##dispatchtypes##_TYPES(self.type(), #op, [&] {            \
      if (self.is_contiguous() && result.is_contiguous()) {                \
        vml::v##op(                                                        \
            result.data<scalar_t>(), self.data<scalar_t>(), self.numel()); \
                                                                           \
      } else {                                                             \
        static constexpr int64_t WIDTH = 131072 / sizeof(scalar_t);        \
        CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(             \
            result,                                                        \
            self,                                                          \
            [](int64_t size,                                               \
               scalar_t* x,                                                \
               scalar_t* y,                                                \
               int64_t stridex,                                            \
               int64_t stridey) {                                          \
              if (stridex == 1 && stridey == 1) {                          \
                vml::v##op(x, y, size);                                    \
              } else {                                                     \
                for (int64_t i = 0; i < size; i += WIDTH) {                \
                  scalar_t buffer[WIDTH];                                  \
                  int64_t width = WIDTH;                                   \
                  width = std::min(width, size - i);                       \
                  for (int64_t j = 0; j < width; j++)                      \
                    buffer[j] = y[stridey * (i + j)];                      \
                  vml::v##op(buffer, buffer, width);                       \
                  for (int64_t j = 0; j < width; j++)                      \
                    x[stridex * (i + j)] = buffer[j];                      \
                }                                                          \
              }                                                            \
            });                                                            \
      }                                                                    \
    });                                                                    \
  }                                                                        \
  REGISTER_DISPATCH(op##Impl, &op##_kernel)

} // anonymous namespace

REGISTER_DISPATCH(sigmoidImpl, &sigmoid_kernel)

// IMPLEMENT_FLOAT_KERNEL(ALL, abs)
IMPLEMENT_FLOAT_KERNEL(FLOATING, acos)
IMPLEMENT_FLOAT_KERNEL(FLOATING, asin)
IMPLEMENT_FLOAT_KERNEL(FLOATING, atan)
IMPLEMENT_FLOAT_KERNEL(FLOATING, ceil)
IMPLEMENT_FLOAT_KERNEL(FLOATING, cos)
// IMPLEMENT_FLOAT_KERNEL(FLOATING, cosh)
IMPLEMENT_FLOAT_KERNEL(FLOATING, erf)
IMPLEMENT_FLOAT_KERNEL(FLOATING, erfc)
IMPLEMENT_FLOAT_KERNEL(FLOATING, exp)
IMPLEMENT_FLOAT_KERNEL(FLOATING, expm1)
IMPLEMENT_FLOAT_KERNEL(FLOATING, floor)
IMPLEMENT_FLOAT_KERNEL(FLOATING, log)
IMPLEMENT_FLOAT_KERNEL(FLOATING, log10)
IMPLEMENT_FLOAT_KERNEL(FLOATING, log1p)
IMPLEMENT_FLOAT_KERNEL(FLOATING, log2)
IMPLEMENT_FLOAT_KERNEL(FLOATING, round)
IMPLEMENT_FLOAT_KERNEL(FLOATING, rsqrt)
IMPLEMENT_FLOAT_KERNEL(FLOATING, sin)
// IMPLEMENT_FLOAT_KERNEL(FLOATING, sinh)
IMPLEMENT_FLOAT_KERNEL(FLOATING, sqrt)
IMPLEMENT_FLOAT_KERNEL(FLOATING, tan)
IMPLEMENT_FLOAT_KERNEL(FLOATING, tanh)
IMPLEMENT_FLOAT_KERNEL(FLOATING, trunc)

}} // namespace at::native
