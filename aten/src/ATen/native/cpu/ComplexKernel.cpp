#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
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
  AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "polar_cpu", [&]() {
    cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
      return c10::complex<scalar_t>(a * std::cos(b), a * std::sin(b));
    });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(complex_stub, &complex_kernel)
ALSO_REGISTER_AVX512_DISPATCH(polar_stub, &polar_kernel)

} // namespace at::native
