#include <ATen/native/ComplexHelper.h>

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

DEFINE_DISPATCH(complex_stub);
DEFINE_DISPATCH(complex_polar_stub);

Tensor& complex_out(Tensor& result, const Tensor& real, const Tensor& imag) {
  TORCH_CHECK(real.scalar_type() == imag.scalar_type(),
              "Expected object of scalar type ", real.scalar_type(), " but got "
              "scalar type ", imag.scalar_type(), " for argument 'imag'");
  // Sort of hacky, but necessary so that input dtypes don't get promoted.
  auto iter = TensorIterator::comparison_op(result, real, imag,
    /*check_mem_overlap=*/true);
  complex_stub(iter.device_type(), iter);
  return result;
}

Tensor complex(const Tensor& real, const Tensor& imag) {
  c10::TensorOptions options = real.options();
  switch (real.scalar_type()) {
    case c10::kFloat: options = options.dtype(c10::kComplexFloat); break;
    case c10::kDouble: options = options.dtype(c10::kComplexDouble); break;
    default: break;
  }
  Tensor result = at::empty(0, options);
  return at::complex_out(result, real, imag);
}

Tensor& complex_polar_out(Tensor& result, const Tensor& abs, const Tensor& angle) {
  TORCH_CHECK(abs.scalar_type() == angle.scalar_type(),
              "Expected object of scalar type ", abs.scalar_type(), " but got "
              "scalar type ", angle.scalar_type(), " for argument 'angle'");
  auto iter = TensorIterator::comparison_op(result, abs, angle,
    /*check_mem_overlap=*/true);
  complex_polar_stub(iter.device_type(), iter);
  return result;
}

Tensor complex_polar(const Tensor& abs, const Tensor& angle) {
  c10::TensorOptions options = abs.options();
  switch (abs.scalar_type()) {
    case c10::kFloat: options = options.dtype(c10::kComplexFloat); break;
    case c10::kDouble: options = options.dtype(c10::kComplexDouble); break;
    default: break;
  }
  Tensor result = at::empty(0, options);
  return at::complex_polar_out(result, abs, angle);
}

}} // namespace at::native
