#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <vector>

namespace at::native {

std::vector<Tensor> _quantize_tensor_cuda(
    const Tensor& /*input*/,
    ScalarType /*qdata_dtype*/,
    int64_t /*inner_scale_calc*/,
    int64_t /*scaling_type*/,
    int64_t /*swizzle_type*/,
    bool /*scaling_type_square_block_and_expand*/) {
  TORCH_CHECK(
      false,
      "torch._quantize_tensor requires a supported NVIDIA GPU and the optional "
      "nvidia-cutlass-dsl and apache-tvm-ffi packages");
  return {};
}

std::vector<Tensor> _quantize_tensor_dual_cuda(
    const Tensor& /*input*/,
    ScalarType /*qdata_dtype*/,
    int64_t /*inner_scale_calc*/,
    int64_t /*scaling_type*/,
    int64_t /*swizzle_type*/,
    bool /*scaling_type_square_block_and_expand*/) {
  TORCH_CHECK(
      false,
      "torch._quantize_tensor_dual requires a supported NVIDIA GPU and the "
      "optional nvidia-cutlass-dsl and apache-tvm-ffi packages");
  return {};
}

}  // namespace at::native
