#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

void gated_delta_rule_cuda_impl(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor decay,
    at::Tensor beta,
    at::Tensor out);

at::Tensor gated_delta_rule_cuda(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor decay,
    at::Tensor beta) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "GDN CUDA kernel expects CUDA tensors");
  TORCH_CHECK(q.dim() == 4, "q must be [B,T,H,K]");
  auto out = at::empty_like(v);
  gated_delta_rule_cuda_impl(
      q.contiguous(),
      k.contiguous(),
      v.contiguous(),
      decay.contiguous(),
      beta.contiguous(),
      out);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gated_delta_rule_cuda", &gated_delta_rule_cuda, "Gated delta rule (CUDA)");
}
