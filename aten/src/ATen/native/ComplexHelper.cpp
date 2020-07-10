#include <ATen/native/ComplexHelper.h>

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

DEFINE_DISPATCH(complex_stub);
DEFINE_DISPATCH(complex_polar_stub);

// expects as input a complex tensor and returns back a tensor
// with corresponding real dtype containing the complex values
// in the last two dimensions
Tensor view_as_real(const Tensor& self) {
  TORCH_CHECK(self.is_complex(), "view_as_real is only supported for complex tensors");
  auto new_sizes = self.sizes().vec();
  // last dimension will always have two elements containing the real and imag vals
  new_sizes.emplace_back(2);
  auto new_strides = computeStrideForViewAsReal(self.strides());
  auto new_storage_offset = 2 * self.storage_offset();
  const auto float_type = c10::toValueType(self.scalar_type());
  return at::empty({0}, self.options().dtype(float_type)).set_(self.storage(), new_storage_offset, new_sizes, new_strides);
}

// expects as input a float or double tensor with last dimension of size 2
// and returns back a tensor with corresponding complex dtype
Tensor view_as_complex(const Tensor& self) {
  TORCH_CHECK(
    self.scalar_type() == kFloat || self.scalar_type() == kDouble,
    "view_as_complex is only supported for float and double tensors, but got a tensor of scalar type: ", self.scalar_type());

  auto new_sizes = self.sizes().vec();
  TORCH_CHECK(new_sizes[self.dim()-1] == 2, "Tensor must have a last dimension of size 2");
  new_sizes.pop_back();

  const auto new_strides = computeStrideForViewAsComplex(self.strides());
  const auto complex_type = c10::toComplexType(self.scalar_type());

  TORCH_CHECK(self.storage_offset() % 2 == 0, "Tensor must have a storage_offset divisible by 2");
  const auto new_storage_offset = self.storage_offset() / 2;

  return at::empty({0}, self.options().dtype(complex_type)).set_(self.storage(), new_storage_offset, new_sizes, new_strides);
}

Tensor& complex_out(Tensor& result, const Tensor& real, const Tensor& imag) {
  // Sort of hacky, but necessary so that input dtypes don't get promoted to complex.
  auto iter = TensorIterator::comparison_op(result, real, imag,
    /*check_mem_overlap=*/true);
  complex_stub(iter.device_type(), iter);
  return result;
}

Tensor complex(const Tensor& real, const Tensor& imag) {
  c10::TensorOptions options = real.options();
  switch (promote_types(real.scalar_type(), imag.scalar_type())) {
    case c10::kDouble: options = options.dtype(c10::kComplexDouble); break;
    default:
      options = options.dtype(c10::kComplexFloat); break;
  }
  Tensor result = at::empty(0, options);
  return at::complex_out(result, real, imag);
}

Tensor& complex_polar_out(Tensor& result, const Tensor& abs, const Tensor& angle) {
  // Sort of hacky, but necessary so that input dtypes don't get promoted to complex.
  auto iter = TensorIterator::comparison_op(result, abs, angle,
    /*check_mem_overlap=*/true);
  complex_polar_stub(iter.device_type(), iter);
  return result;
}

Tensor complex_polar(const Tensor& abs, const Tensor& angle) {
  c10::TensorOptions options = abs.options();
  switch (promote_types(abs.scalar_type(), angle.scalar_type())) {
    case c10::kDouble: options = options.dtype(c10::kComplexDouble); break;
    default:
      options = options.dtype(c10::kComplexFloat); break;
  }
  Tensor result = at::empty(0, options);
  return at::complex_polar_out(result, abs, angle);
}

}} // namespace at::native
