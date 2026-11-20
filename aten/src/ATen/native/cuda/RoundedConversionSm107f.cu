#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/RoundedConversion.cuh>

#include <ATen/Dispatch_v2.h>

namespace at::native {

void to_dtype_rounded_kernel_cuda_sm107f(
    TensorIteratorBase& iter,
    c10::RoundingMode mode) {
  AT_DISPATCH_V2(
      iter.dtype(),
      "to_dtype_rounded_cuda_sm107f",
      AT_WRAP([&] { to_dtype_rounded_kernel_cuda_to<scalar_t>(iter, mode); }),
      kFloat8_e4m3fn,
      kFloat8_e5m2,
      kFloat8_e5m3fnu);
}

} // namespace at::native
