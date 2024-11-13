#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/BetaOps.h>

#include <ATen/Dispatch.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/TypeSafeSignMath.h>

namespace at::native {

namespace {

void betainc_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "betainc_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t x, scalar_t a, scalar_t b) -> scalar_t {
        return calc_betainc(x, a, b);
    });
  });
}

} // namespace

REGISTER_DISPATCH(betainc_stub, &betainc_kernel)

} // namespace at::native
