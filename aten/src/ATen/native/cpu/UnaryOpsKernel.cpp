#include "ATen/native/cpu/UnaryOpsKernel.h"

#include <cmath>
#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/cpu/vec256/vec256.h"
#include "ATen/cpu/vec256/functional.h"
#include "ATen/native/cpu/CapabilityDispatch.h"

namespace at { namespace native {
namespace {

using namespace vec256;

static void abs_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_ALL_TYPES(self.type(), "abs", [&] {
    scalar_t* out = result.data<scalar_t>();
    scalar_t* in = self.data<scalar_t>();
    int64_t size = self.numel();
    parallel_for(0, size, 2048, [out, in](int64_t begin, int64_t end) {
      map([](const Vec256<scalar_t>& x) { return x.abs(); },
          out + begin,
          in + begin,
          end - begin);
    });
  });
}

static void rsqrt_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "rsqrt", [&] {
    scalar_t* out = result.data<scalar_t>();
    scalar_t* in = self.data<scalar_t>();
    int64_t size = self.numel();
    parallel_for(0, size, 2048, [out, in](int64_t begin, int64_t end) {
      map(
          [](const Vec256<scalar_t>& x) {
            return Vec256<scalar_t>((scalar_t)(1)) / x.sqrt();
          },
          out + begin,
          in + begin,
          end - begin);
    });
  });
}

#define IMPLEMENT_FLOAT_KERNEL(op)                                        \
  static void op##_kernel(Tensor& result, const Tensor& self) {           \
    AT_DISPATCH_FLOATING_TYPES(self.type(), #op, [&] {                    \
      scalar_t* out = result.data<scalar_t>();                            \
      scalar_t* in = self.data<scalar_t>();                               \
      int64_t size = self.numel();                                        \
      parallel_for(0, size, 2048, [out, in](int64_t begin, int64_t end) { \
        map([](const Vec256<scalar_t>& x) { return x.op(); },             \
            out + begin,                                                  \
            in + begin,                                                   \
            end - begin);                                                 \
      });                                                                 \
    });                                                                   \
  }                                                                       \
  REGISTER_DISPATCH(op##Impl, &op##_kernel)

} // anonymous namespace


REGISTER_DISPATCH(absImpl, &abs_kernel);
REGISTER_DISPATCH(rsqrtImpl, &rsqrt_kernel);

IMPLEMENT_FLOAT_KERNEL(acos)
IMPLEMENT_FLOAT_KERNEL(asin)
IMPLEMENT_FLOAT_KERNEL(atan)
IMPLEMENT_FLOAT_KERNEL(erf)
IMPLEMENT_FLOAT_KERNEL(exp)
IMPLEMENT_FLOAT_KERNEL(expm1)
IMPLEMENT_FLOAT_KERNEL(log)
IMPLEMENT_FLOAT_KERNEL(log10)
IMPLEMENT_FLOAT_KERNEL(log1p)
IMPLEMENT_FLOAT_KERNEL(log2)
IMPLEMENT_FLOAT_KERNEL(ceil)
IMPLEMENT_FLOAT_KERNEL(floor)
IMPLEMENT_FLOAT_KERNEL(round)
IMPLEMENT_FLOAT_KERNEL(sqrt)
IMPLEMENT_FLOAT_KERNEL(tanh)
IMPLEMENT_FLOAT_KERNEL(trunc)

}} // namespace at::native
