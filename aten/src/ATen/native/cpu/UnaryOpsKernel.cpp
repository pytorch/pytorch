#include "ATen/native/cpu/UnaryOpsKernel.h"

#include <cmath>
#include "ATen/Dispatch.h"
#include "ATen/cpu/vml.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/native/cpu/CapabilityDispatch.h"

namespace at { namespace native {
namespace {

using namespace vec256;

#define IMPLEMENT_FLOAT_KERNEL(dispatchtypes, op)                          \
  static void op##_kernel(Tensor& result, const Tensor& self) {            \
    AT_DISPATCH_##dispatchtypes##_TYPES(self.type(), #op, [&] {                \
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

// IMPLEMENT_FLOAT_KERNEL(ALL, abs)
IMPLEMENT_FLOAT_KERNEL(FLOATING, acos)
IMPLEMENT_FLOAT_KERNEL(FLOATING, asin)
IMPLEMENT_FLOAT_KERNEL(FLOATING, atan)
IMPLEMENT_FLOAT_KERNEL(FLOATING, ceil)
IMPLEMENT_FLOAT_KERNEL(FLOATING, cos)
// IMPLEMENT_FLOAT_KERNEL(FLOATING, cosh)
IMPLEMENT_FLOAT_KERNEL(FLOATING, erf)
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
