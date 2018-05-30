#include "ATen/native/cpu/UnaryOpsKernel.h"

#include <cmath>
#include <iostream>
#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/cpu/vec256/vec256.h"
#include "ATen/cpu/vec256/functional.h"
#include "ATen/native/cpu/CapabilityDispatch.h"

namespace at { namespace native {
namespace {

using namespace vec256;

template <class scalar_t, class F>
static void parallel_apply(Tensor& result, const Tensor& self, F f) {
  auto arr_out = result.data<scalar_t>();
  auto arr_in = self.data<scalar_t>();
  int64_t size = self.numel();
  parallel_for_1d(
      [arr_out, arr_in, f](int64_t begin, int64_t end) {
        map(f, arr_out + begin, arr_in + begin, end - begin);
      },
      0,
      size,
      1024);
}

static void abs_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_ALL_TYPES(self.type(), "abs", [&] {
    parallel_apply<scalar_t>(
        result,
        self,
        [](const Vec256<scalar_t>& x) { return x.abs(); });  });
}

static void rsqrt_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "rsqrt", [&] {
    parallel_apply<scalar_t>(
        result,
        self,
        [](const Vec256<scalar_t>& x) { return Vec256<scalar_t>((scalar_t)(1)) / x.sqrt(); });  });
}

#define IMPLEMENT_FLOAT_KERNEL(op)                                             \
  static void op##_kernel(Tensor& result, const Tensor& self) {                \
    AT_DISPATCH_FLOATING_TYPES(self.type(), #op, [&] {                         \
      parallel_apply<scalar_t>(                                                \
          result, self, [](const Vec256<scalar_t>& x) { return x.op(); }); \
    });                                                                        \
  }                                                                            \
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
