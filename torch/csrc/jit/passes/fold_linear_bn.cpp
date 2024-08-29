#include <torch/csrc/jit/passes/fold_linear_bn.h>

#include <ATen/TensorOperators.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/rsqrt.h>
#endif

namespace torch::jit {

std::tuple<at::Tensor, at::Tensor> computeUpdatedLinearWeightAndBias(
    const LinearBNParameters& p) {
  at::Tensor bn_scale = p.bn_w * at::rsqrt(p.bn_rv + p.bn_eps);
  at::Tensor fused_w = p.linear_w * bn_scale.unsqueeze(-1);
  at::Tensor fused_b = (p.linear_b - p.bn_rm) * bn_scale + p.bn_b;

  auto linear_w_dtype = p.linear_w.dtype();
  auto linear_b_dtype = p.linear_b.dtype();

  return std::make_tuple(
      fused_w.to(linear_w_dtype), fused_b.to(linear_b_dtype));
}

} // namespace torch::jit
