#include <ATen/native/cuda/RowwiseScaledMM.h>
#include <ATen/native/cuda/RowwiseScaledMMSM89.h>
#include <ATen/native/cuda/RowwiseScaledMMSM9x.h>
#include <ATen/native/cuda/RowwiseScaledMMSM10x.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/macros/Macros.h>

namespace at::cuda::detail {
void f8f8bf16_rowwise(
    at::Tensor XQ, // FP8
    at::Tensor WQ, // FP8
    at::Tensor x_scale, // FP32
    at::Tensor w_scale, // FP32
    std::optional<at::Tensor> bias, // BF16
    bool use_fast_accum,
    at::Tensor& out) {
  cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  const bool sm89 = properties != nullptr && properties->major == 8 && properties->minor == 9;
  const bool sm9x = properties != nullptr && properties->major == 9;
  const bool sm10x = properties != nullptr && properties->major == 10;

  if (sm89) {
    f8f8bf16_rowwise_sm89(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out);
  } else if (sm9x) {
    f8f8bf16_rowwise_sm9x(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out);
  } else if (sm10x) {
    f8f8bf16_rowwise_sm10x(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out);
  } else {
    TORCH_CHECK(
        false, "Rowwise scaling is not currently supported on your device");
  }
}
} // namespace at::cuda::detail
