#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/RoundedConversion.cuh>

#include <ATen/Dispatch_v2.h>

namespace at::native {
namespace {

void to_dtype_rounded_kernel_cuda(
    TensorIteratorBase& iter,
    c10::RoundingMode mode) {
  if (iter.dtype() == kFloat8_e8m0fnu || iter.dtype() == kFloat8_e5m3fnu) {
    to_dtype_rounded_kernel_cuda_arch_specific(iter, mode);
    return;
  }
  AT_DISPATCH_V2(
      iter.input_dtype(),
      "to_dtype_rounded_cuda",
      AT_WRAP([&] {
        using src_t = scalar_t;
        AT_DISPATCH_V2(
            iter.dtype(),
            "to_dtype_rounded_cuda",
            AT_WRAP([&] { to_dtype_rounded_loop<src_t, scalar_t>(iter, mode); }),
            kFloat,
            kHalf,
            kBFloat16,
            kFloat8_e4m3fn,
            kFloat8_e5m2,
            kFloat8_e4m3fnuz,
            kFloat8_e5m2fnuz);
      }),
      kDouble,
      kFloat,
      kHalf,
      kBFloat16);
}

} // namespace

REGISTER_DISPATCH(to_dtype_rounded_stub, &to_dtype_rounded_kernel_cuda)

} // namespace at::native
