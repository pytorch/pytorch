#include <c10/util/Exception.h>
#include <c10/util/ArrayRef.h>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/utils/ParamsHash.h>
#include <ATen/NativeFunctions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/library.h>

namespace at {
namespace native {

Tensor gelu_quantized_cuda(const Tensor& qx, c10::string_view approximate) {
  if (qx.numel() == 0) {
    return Tensor{};
  }
  TORCH_CHECK(qx.qscheme() == at::kPerTensorAffine, "gelu_quantized_cuda only supports per tensor quantized tensors");
  auto x_fp32 = at::dequantize(qx);
  auto result_fp32 = at::gelu(x_fp32);
  return at::quantize_per_tensor(result_fp32, qx.q_scale(), qx.q_zero_point(), qx.scalar_type());
}

}  // namespace at::native
}  // namespace at
