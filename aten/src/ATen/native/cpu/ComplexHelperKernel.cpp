#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/ComplexHelper.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

namespace at { namespace native {
namespace {

using namespace vec256;

void complex_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "complex_cpu", [&]() {
    cpu_kernel(iter,
      [=](scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
        return c10::complex<scalar_t>(a, b);
      });
  });
}

void complex_polar_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "complex_polar_cpu", [&]() {
    cpu_kernel(iter,
      [=](scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
        return c10::complex<scalar_t>(a * std::cos(b), a * std::sin(b));
      });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(complex_stub, &complex_kernel);
REGISTER_DISPATCH(complex_polar_stub, &complex_polar_kernel);

}} // namespace at::native