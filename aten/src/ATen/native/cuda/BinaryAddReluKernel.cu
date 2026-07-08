#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/core/Scalar.h>

namespace at::native {

void add_clamp_kernel_cuda(
    TensorIterator& iter,
    const Scalar& alpha_scalar,
    const Scalar& min_val,
    const Scalar& max_val) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, iter.dtype(), "add_clamp_cuda", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;

        opmath_t alpha{alpha_scalar.to<opmath_t>()};
        opmath_t lo{min_val.to<opmath_t>()};
        opmath_t hi{max_val.to<opmath_t>()};

        gpu_kernel(
            iter, [alpha, lo, hi] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
              opmath_t r(static_cast<opmath_t>(a) + alpha * static_cast<opmath_t>(b));

              // clamp to [lo, hi] using comparisons that preserve NaNs.
              opmath_t m = (r < lo) ? lo : r;
              m = (m > hi) ? hi : m;
              return static_cast<scalar_t>(m);
            });
      });
}

REGISTER_DISPATCH(add_clamp_stub, &add_clamp_kernel_cuda)

} // namespace at::native
