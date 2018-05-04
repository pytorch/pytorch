#include "ATen/native/cpu/UnaryOpsKernel.h"

#include <cmath>
#include <iostream>
#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/cpu/vec256/vec256.h"
#include "ATen/native/cpu/CapabilityDispatch.h"

namespace at { namespace native {
namespace {

using namespace vec256;

template <typename scalar_t, typename F>
static void
unary_kernel(scalar_t* arr_out, const scalar_t* arr_in, int64_t size, F func) {
  using Vec = Vec256<scalar_t>;
  int64_t size_rounded = size - (size % Vec::size);
  int64_t k = 0;
  for (; k != size_rounded; k += Vec::size) {
    auto value = func(Vec::s_load(arr_in + k));
    value.store(arr_out + k);
  }
  auto leftover = size - k;
  if (leftover > 0) {
    Vec a;
    a.load_partial(arr_in + k, leftover);
    func(a).store_partial(arr_out + k, leftover);
  }
}

template <class scalar_t, class F>
static void parallel_apply(Tensor& result, const Tensor& self, F f) {
  internal::init_tbb_num_threads();

  static tbb::affinity_partitioner ap;

  auto arr_out = result.data<scalar_t>();
  auto arr_in = self.data<scalar_t>();
  int64_t size = self.numel();
  if (size < internal::TBB_GRAIN_SIZE) {
    unary_kernel(arr_out, arr_in, size, f);
  } else {
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, size, internal::TBB_GRAIN_SIZE),
        [&](const tbb::blocked_range<int64_t>& r) {
          auto size = r.end() - r.begin();
          unary_kernel(arr_out + r.begin(), arr_in + r.begin(), size, f);
        },
        ap);
  }
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
IMPLEMENT_FLOAT_KERNEL(trunc)

}} // namespace at::native
