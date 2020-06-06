#include <ATen/native/ComplexHelper.h>

namespace at {
namespace native {

Tensor& complex_out(Tensor& result, const Tensor& input1, const Tensor& input2, bool polar) {
  TORCH_CHECK(!input1.is_complex(), "input1 should not be a complex tensor.");
  TORCH_CHECK(!input2.is_complex(), "input2 should not be a complex tensor.");

  c10::Scalar i;
  switch (result.scalar_type()) {
    // case c10::kComplexHalf:
    //   i = c10::complex<c10::Half>(0, 1);
    case c10::kComplexFloat:
      i = c10::complex<float>(0, 1);
      break;
    case c10::kComplexDouble:
      i = c10::complex<double>(0, 1);
      break;
    default:
      break;
  }

  if (!polar) {
    at::add_out(result, input1, at::mul(input2, i));
  } else {
    Tensor real = at::mul(input1, at::cos(input2));
    Tensor imag = at::mul(input1, at::sin(input2));
    at::add_out(result, real, at::mul(imag, i));
  }
  return result;
}

Tensor complex(const Tensor& input1, const Tensor& input2, bool polar) {

  c10::ScalarType result_type = promote_types(input1.scalar_type(), input2.scalar_type());
  c10::TensorOptions options = input1.options();
  switch (result_type) {
    case c10::kShort:
      options = options.dtype(c10::kComplexHalf);
      break;
    case c10::kInt:
      options = options.dtype(c10::kComplexFloat);
      break;
    case c10::kLong:
      options = options.dtype(c10::kComplexDouble);
      break;
    case c10::kHalf:
      options = options.dtype(c10::kComplexHalf);
      break;
    case c10::kFloat:
      options = options.dtype(c10::kComplexFloat);
      break;
    case c10::kDouble:
      options = options.dtype(c10::kComplexDouble);
      break;
    default:
      options = options.dtype(c10::kComplexDouble);
      break;
  }
  Tensor result = at::empty(0, options);
  return at::complex_out(result, input1, input2, polar);
}

}} // namespace at::native
