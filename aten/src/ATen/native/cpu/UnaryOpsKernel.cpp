#include "ATen/native/cpu/UnaryOpsKernel.h"

#include <cmath>
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/cpu/vec256/functional.h"
#include "ATen/cpu/vec256/vec256.h"
#include "ATen/native/cpu/CapabilityDispatch.h"

namespace at { namespace native {
namespace {

using namespace vec256;

static void abs_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_ALL_TYPES(self.type(), "abs", [&] {
    CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(
        result,
        self,
        [](int64_t size,
           scalar_t* x,
           scalar_t* y,
           int64_t stridex,
           int64_t stridey) {
          if (stridex == 1 && stridey == 1) {
            map([](const Vec256<scalar_t>& x) { return x.abs(); }, x, y, size);
          } else {
            for (int64_t i = 0; i < size; i++) {
              x[stridex * i] = std::abs(y[stridey * i]);
            }
          }
        });
  });
}

// [Note AVX-SSE transitions] In general we avoid calls into cmath for code
// compiled with AVX/AVX2 This is because of SSE-AVX transitions and a bug in
// Glibc2.23 See https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/1663280
// Calling zeroupper when using AVX/AVX2 code resolves this.
#if defined(__AVX__) && defined(__GLIBC__) && __GLIBC_MINOR__ == 23
#define ZEROUPPER _mm256_zeroupper();
#else
#define ZEROUPPER
#endif

#define IMPLEMENT_FLOAT_COMPUTEBOUND_KERNEL(op, opfn)                   \
  static void op##_kernel(Tensor& result, const Tensor& self) {         \
    AT_DISPATCH_FLOATING_TYPES(self.type(), #op, [&] {                  \
      static constexpr int WIDTH = 128 / sizeof(scalar_t);              \
      CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(            \
          result,                                                       \
          self,                                                         \
          [](int64_t size,                                              \
             scalar_t* x,                                               \
             scalar_t* y,                                               \
             int64_t stridex,                                           \
             int64_t stridey) {                                         \
            if (stridex == 1 && stridey == 1) {                         \
              map([](const Vec256<scalar_t>& x) { return x.op(); },     \
                  x,                                                    \
                  y,                                                    \
                  size);                                                \
            } else {                                                    \
              int64_t i = 0;                                            \
              if (size > WIDTH) {                                       \
                for (; i < size - size % WIDTH; i += WIDTH) {           \
                  scalar_t buffer_in[WIDTH];                            \
                  scalar_t buffer_out[WIDTH];                           \
                  for (int64_t j = 0; j < WIDTH; j++)                   \
                    buffer_in[j] = y[stridey * (j + i)];                \
                  map([](const Vec256<scalar_t>& x) { return x.op(); }, \
                      buffer_out,                                       \
                      buffer_in,                                        \
                      WIDTH);                                           \
                  for (int64_t j = 0; j < WIDTH; j++)                   \
                    x[stridex * (j + i)] = buffer_out[j];               \
                }                                                       \
              }                                                         \
              for (; i < size; i++) {                                   \
                ZEROUPPER                                               \
                x[stridex * i] = opfn(y[stridey * i]);                  \
              }                                                         \
            }                                                           \
          });                                                           \
    });                                                                 \
  }                                                                     \
  REGISTER_DISPATCH(op##Impl, &op##_kernel)

#define IMPLEMENT_FLOAT_KERNEL(op, opfn)                            \
  static void op##_kernel(Tensor& result, const Tensor& self) {     \
    AT_DISPATCH_FLOATING_TYPES(self.type(), #op, [&] {              \
      CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(        \
          result,                                                   \
          self,                                                     \
          [](int64_t size,                                          \
             scalar_t* x,                                           \
             scalar_t* y,                                           \
             int64_t stridex,                                       \
             int64_t stridey) {                                     \
            if (stridex == 1 && stridey == 1) {                     \
              map([](const Vec256<scalar_t>& x) { return x.op(); }, \
                  x,                                                \
                  y,                                                \
                  size);                                            \
            } else {                                                \
              for (int64_t i = 0; i < size; i++) {                  \
                ZEROUPPER                                           \
                x[stridex * i] = opfn(y[stridey * i]);              \
              }                                                     \
            }                                                       \
          });                                                       \
    });                                                             \
  }                                                                 \
  REGISTER_DISPATCH(op##Impl, &op##_kernel)

#define IMPLEMENT_FLOAT_KERNEL_LOOP(op, opfn)                   \
  static void op##_kernel(Tensor& result, const Tensor& self) { \
    AT_DISPATCH_FLOATING_TYPES(self.type(), #op, [&] {          \
      CPU_tensor_parallel_kernel_apply2<scalar_t, scalar_t>(    \
          result,                                               \
          self,                                                 \
          [](int64_t size,                                      \
             scalar_t* x,                                       \
             scalar_t* y,                                       \
             int64_t stridex,                                   \
             int64_t stridey) {                                 \
            for (int64_t i = 0; i < size; i++) {                \
              ZEROUPPER                                         \
              x[stridex * i] = opfn(y[stridey * i]);            \
            }                                                   \
          });                                                   \
    });                                                         \
  }                                                             \
  REGISTER_DISPATCH(op##Impl, &op##_kernel)

} // anonymous namespace

REGISTER_DISPATCH(absImpl, &abs_kernel);

IMPLEMENT_FLOAT_COMPUTEBOUND_KERNEL(acos, std::acos)
IMPLEMENT_FLOAT_COMPUTEBOUND_KERNEL(asin, std::asin)
IMPLEMENT_FLOAT_COMPUTEBOUND_KERNEL(atan, std::atan)
IMPLEMENT_FLOAT_KERNEL(ceil, std::ceil)
IMPLEMENT_FLOAT_KERNEL(erf, std::erf)
IMPLEMENT_FLOAT_COMPUTEBOUND_KERNEL(exp, std::exp)
IMPLEMENT_FLOAT_COMPUTEBOUND_KERNEL(expm1, std::expm1)
IMPLEMENT_FLOAT_KERNEL(floor, std::floor)
IMPLEMENT_FLOAT_COMPUTEBOUND_KERNEL(log, std::log)
IMPLEMENT_FLOAT_COMPUTEBOUND_KERNEL(log10, std::log10)
IMPLEMENT_FLOAT_COMPUTEBOUND_KERNEL(log1p, std::log1p)
IMPLEMENT_FLOAT_KERNEL(log2, std::log2)
IMPLEMENT_FLOAT_KERNEL(round, std::round)
IMPLEMENT_FLOAT_COMPUTEBOUND_KERNEL(rsqrt, 1 / std::sqrt)
IMPLEMENT_FLOAT_COMPUTEBOUND_KERNEL(sqrt, std::sqrt)
IMPLEMENT_FLOAT_COMPUTEBOUND_KERNEL(tanh, std::tanh)
IMPLEMENT_FLOAT_KERNEL(trunc, std::trunc)

IMPLEMENT_FLOAT_KERNEL_LOOP(cos, std::cos)
IMPLEMENT_FLOAT_KERNEL_LOOP(cosh, std::cosh)
IMPLEMENT_FLOAT_KERNEL_LOOP(sin, std::sin)
IMPLEMENT_FLOAT_KERNEL_LOOP(sinh, std::sinh)
IMPLEMENT_FLOAT_KERNEL_LOOP(tan, std::tan)

}} // namespace at::native
