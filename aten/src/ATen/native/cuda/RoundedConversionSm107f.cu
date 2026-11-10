#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/RoundedConversion.cuh>

namespace at::native {

void to_dtype_rounded_kernel_cuda_sm107f(
    TensorIteratorBase& iter,
    c10::RoundingMode mode) {
  switch (iter.dtype()) {
    case kFloat8_e4m3fn:
      to_dtype_rounded_kernel_cuda_to<Float8_e4m3fn>(iter, mode);
      break;
    case kFloat8_e5m2:
      to_dtype_rounded_kernel_cuda_to<Float8_e5m2>(iter, mode);
      break;
    case kFloat8_e5m3fnu:
      to_dtype_rounded_kernel_cuda_to<Float8_e5m3fnu>(iter, mode);
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "unexpected dtype ", iter.dtype());
  }
}

} // namespace at::native
