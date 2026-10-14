#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <vector>

namespace at::native {

std::vector<Tensor> _quantize_tensor_cuda(
    const Tensor& /*input*/,
    int64_t /*scaling_type*/,
    ScalarType /*qdata_dtype*/,
    int64_t /*scaling_algorithm*/,
    int64_t /*swizzle_type*/,
    bool /*scaling_type_use_square_block_size*/) {
  TORCH_CHECK(
      false,
      "torch._quantize_tensor requires an NVIDIA GPU with CUDA compute "
      "capability 10.0 or higher and the optional nvidia-cutlass-dsl and "
      "apache-tvm-ffi packages");
  return {};
}

std::vector<Tensor> _quantize_tensor_dual_cuda(
    const Tensor& /*input*/,
    int64_t /*scaling_type*/,
    ScalarType /*qdata_dtype*/,
    int64_t /*scaling_algorithm*/,
    int64_t /*swizzle_type*/,
    bool /*scaling_type_use_square_block_size*/) {
  TORCH_CHECK(
      false,
      "torch._quantize_tensor_dual requires an NVIDIA GPU with CUDA compute "
      "capability 10.0 or higher and the optional nvidia-cutlass-dsl and "
      "apache-tvm-ffi packages");
  return {};
}

}  // namespace at::native
