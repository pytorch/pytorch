#include <ATen/native/cuda/RowwiseScaledMMSM89.h>
#include <ATen/native/cuda/RowwiseScaledMM.cuh>

void f8f8bf16_rowwise_sm89(at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale,
                           at::Tensor w_scale, std::optional<at::Tensor> bias,
                           bool use_fast_accum, at::Tensor& out) {
#if defined(BUILD_ROWWISE_FP8_KERNEL)
  f8f8bf16_rowwise_impl<cutlass::arch::Sm89>(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out);
#else
  TORCH_CHECK(
    false, "Rowwise scaling is not currently supported on your device");
#endif
}
