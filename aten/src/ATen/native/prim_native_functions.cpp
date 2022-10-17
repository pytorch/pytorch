#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/is_nonzero_native.h>
#endif

namespace at {
namespace native {

bool is_nonzero(const Tensor& self) {
  auto n = self.numel();
  TORCH_CHECK(n != 0, "Boolean value of Tensor with no values is ambiguous");
  TORCH_CHECK(
      n < 2, "Boolean value of Tensor with more than one value is ambiguous");

  Scalar localScalar = self.item();
  if (localScalar.isFloatingPoint()) {
    return localScalar.to<double>() != 0;
  } else if (localScalar.isComplex()) {
    return localScalar.to<c10::complex<double>>() !=
        c10::complex<double>(0.0, 0.0);
  } else if (localScalar.isIntegral(false)) {
    return localScalar.to<int64_t>() != 0;
  } else if (localScalar.isBoolean()) {
    return localScalar.to<bool>();
  }
  TORCH_INTERNAL_ASSERT(false, "Expected non-Tensor backend scalar");
}


// Aux function used in the test TestPythonDispatch.test_kwarg_only_and_positional_default
// within test/test_python_dispatch.py
Tensor foobar(const Tensor& self, bool arg1, bool arg2, bool arg3) {
  return self;
}

} // namespace meta
} // namespace at
