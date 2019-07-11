#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#include <ATen/native/Fill.h>

namespace at { namespace native {
namespace {

void fill_kernel(TensorIterator& iter, Scalar value_scalar) {
  if (iter.dtype() == ScalarType::Half) {
    auto value = value_scalar.to<at::Half>().x;
    using H = decltype(value);
    cpu_kernel_vec(
        iter,
        [=]() -> H { return value; },
        [=]() { return Vec256<H>(value); });
  } else if (iter.dtype() == ScalarType::BFloat16) {
    auto value = value_scalar.to<at::BFloat16>().x;
    using H = decltype(value);
    cpu_kernel_vec(
        iter,
        [=]() -> H { return value; },
        [=]() { return Vec256<H>(value); });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool, iter.dtype(), "fill_cpu", [&]() {
      scalar_t value = value_scalar.to<scalar_t>();
      cpu_kernel_vec(
          iter,
          [=]() -> scalar_t { return value; },
          [=]() { return Vec256<scalar_t>(value); });
    });
  }
}

} // namespace

REGISTER_DISPATCH(fill_stub, &fill_kernel);

} // namespace native
} // namespace at
