#include <ATen/native/cuda/RowwiseScaledMMSM10x.h>
#include <ATen/native/cuda/RowwiseScaledMM.cuh>

void f8f8bf16_rowwise_sm10x(at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale,
                            at::Tensor w_scale, std::optional<at::Tensor> bias,
                            bool use_fast_accum, at::Tensor& out) {
  f8f8bf16_rowwise_impl<cutlass::arch::Sm100>(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, out);
}
