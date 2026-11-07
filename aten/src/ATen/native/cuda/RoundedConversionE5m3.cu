#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/RoundedConversion.cuh>

namespace at::native {

void to_dtype_rounded_kernel_cuda_e5m3(
    TensorIteratorBase& iter,
    c10::RoundingMode mode) {
  to_dtype_rounded_kernel_cuda_to<Float8_e5m3fnu>(iter, mode);
}

} // namespace at::native
