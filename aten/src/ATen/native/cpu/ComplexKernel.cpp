#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at::native {
namespace {

void complex_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.input_dtype(), "complex_cpu", [&]() {
    cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
      return c10::complex<scalar_t>(a, b);
    });
  });
}

void polar_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      kHalf, iter.input_dtype(), "polar_cpu", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
          return c10::complex<scalar_t>(
              static_cast<scalar_t>(opmath_t(a) * std::cos(opmath_t(b))),
              static_cast<scalar_t>(opmath_t(a) * std::sin(opmath_t(b))));
        });
      });
}

} // anonymous namespace

REGISTER_DISPATCH(complex_stub, &complex_kernel)
ALSO_REGISTER_AVX512_DISPATCH(polar_stub, &polar_kernel)

} // namespace at::native
