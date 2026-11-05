#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/RoundedConversion.cuh>

#include <ATen/Dispatch_v2.h>

namespace at::native {

void to_dtype_rounded_kernel_cuda_arch_specific(
    TensorIteratorBase& iter,
    c10::RoundingMode mode) {
  AT_DISPATCH_V2(
      iter.input_dtype(),
      "to_dtype_rounded_cuda",
      AT_WRAP([&] {
        using src_t = scalar_t;
        AT_DISPATCH_V2(
            iter.dtype(),
            "to_dtype_rounded_cuda",
            AT_WRAP([&] { to_dtype_rounded_loop<src_t, scalar_t>(iter, mode); }),
            kFloat8_e8m0fnu,
            kFloat8_e5m3fnu);
      }),
      kDouble,
      kFloat,
      kHalf,
      kBFloat16);
}

} // namespace at::native
