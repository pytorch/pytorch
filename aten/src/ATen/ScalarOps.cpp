// FastPass
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#endif

#include <ATen/ScalarOps.h>
#include <ATen/ATen.h>
#include <ATen/Utils.h>

namespace at {
namespace {
template <typename scalar_t>
inline void fill_inplace(Tensor& self, const Scalar& value_scalar) {
  auto value = value_scalar.to<scalar_t>();
  scalar_t* dptr = static_cast<scalar_t*>(self.data_ptr());
  *dptr = value;
}
}

namespace detail {
Tensor& scalar_fill(Tensor& self, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBool, kBFloat16, self.scalar_type(), "fill_out", [&]() {
        fill_inplace<scalar_t>(self, value);
      });
  return self;
}

Tensor scalar_tensor_static(const Scalar& s, c10::optional<ScalarType> dtype_opt, c10::optional<Device> device_opt) {
  at::tracer::impl::NoTracerDispatchMode tracer_guard;
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  auto result = at::detail::empty_cpu({}, dtype_opt, c10::nullopt, device_opt, c10::nullopt, c10::nullopt);
  scalar_fill(result, s);
  return result;
}
} // namespace detail
} // namespace at
