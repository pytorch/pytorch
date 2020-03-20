#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#include <ATen/native/Fill.h>

namespace at { namespace native {
namespace {


template <typename scalar_t>
static void fill_non_native_type(TensorIterator& iter, Scalar value_scalar) {
  auto value = value_scalar.to<scalar_t>().x;
  using H = typename std::make_signed<decltype(value)>::type;  // Signed type has more acceleration
  // Reserve the representation of value. static_cast<H>(value) is implementation defined.
  H val = *reinterpret_cast<H*>(std::addressof(value));
  cpu_kernel_vec(
      iter,
      [val]() -> H { return val; },
      [val]() { return Vec256<H>(val); });
}

void fill_kernel(TensorIterator& iter, Scalar value_scalar) {
  if (iter.dtype() == ScalarType::Half) {
    fill_non_native_type<at::Half>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::BFloat16) {
    fill_non_native_type<at::BFloat16>(iter, value_scalar);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Bool, iter.dtype(), "fill_cpu", [&]() {
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
