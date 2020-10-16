// FastPass
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#endif

#include <ATen/FastPass.h>

namespace at {
namespace {
  template <typename scalar_t>
  inline void fill_fast(Tensor& self, Scalar value_scalar) {
    auto value = value_scalar.to<scalar_t>();
    scalar_t * dptr = static_cast<scalar_t *>(self.data_ptr());
    *dptr = value;
  }
  Tensor fast_pass_fill(Tensor& self, Scalar value) {
      AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, self.scalar_type(), "fill_out", [&]() {
          fill_fast<scalar_t>(self, value);});
      return self;
  }
} // namespace

Tensor scalar_tensor_fast(Scalar s, const TensorOptions& options) {
  TORCH_CHECK(options.device() == kCPU);
  at::tracer::impl::NoTracerDispatchMode tracer_guard;
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  auto result = native::empty_cpu({}, options);
  fast_pass_fill(result, s);
  return result;
}
} //namespace at
