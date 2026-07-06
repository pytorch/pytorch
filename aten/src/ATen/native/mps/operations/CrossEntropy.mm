#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/CrossEntropyKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_cross_entropy_loss_2d_backward_native.h>
#include <ATen/ops/_fused_cross_entropy_loss_2d_forward_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/sum.h>
#endif

namespace at::native {

namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& ce_lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/CrossEntropyKernel_metallib.h>
static auto& ce_lib = lib;
#endif

// Returns the per-row threadgroup size. Each thread reads N_READS=4 logits, so
// at most ceil(V / 4) threads have any work; launching more than that just
// burns idle threads (and full-width simd/barrier overhead) on every one of
// the B threadgroups. That dominates for many-rows/few-classes shapes
// (e.g. [131072, 10]), where a fixed 1024-wide TG ran ~5-9x slower than the
// reference. We therefore size the TG to the work, rounded UP to a whole
// number of execution widths, then cap at min(maxTotalThreadsPerThreadgroup,
// 1024) so the online reduction still fits one threadgroup for large V.
static uint32_t ce_threadgroup_size(id<MTLComputePipelineState> pso, int64_t V) {
  uint32_t exec_w = static_cast<uint32_t>([pso threadExecutionWidth]);
  uint32_t cap = std::min<uint32_t>(1024u, static_cast<uint32_t>([pso maxTotalThreadsPerThreadgroup]));
  cap = std::max(exec_w, (cap / exec_w) * exec_w);

  // threads actually needed: ceil(V / N_READS), N_READS == 4 in the kernel.
  constexpr uint32_t N_READS = 4;
  uint64_t needed = (static_cast<uint64_t>(V) + N_READS - 1) / N_READS;
  // round up to a whole number of execution widths (>= one width).
  uint64_t rounded = ((needed + exec_w - 1) / exec_w) * exec_w;
  uint32_t tg_size = static_cast<uint32_t>(std::min<uint64_t>(rounded, cap));
  return std::max(exec_w, tg_size);
}

} // namespace mps

// Forward: per-row fused cross-entropy. Returns
//   loss[B]           : unreduced per-row loss (fp32),
//   lse[B]            : per-row log-sum-exp saved for backward (fp32),
//   weight_per_row[B] : w[target] (or 1; 0 when ignored), for the host-side
//                       mean normalizer.
// The host applies the {mean,sum,none} reduction with plain differentiable
// aten ops so autograd stays a standard composite (see cross_entropy_loss).
std::tuple<Tensor, Tensor, Tensor> _fused_cross_entropy_loss_2d_forward_mps(const Tensor& self,
                                                                            const Tensor& target,
                                                                            const std::optional<Tensor>& weight_opt,
                                                                            int64_t ignore_index,
                                                                            double label_smoothing) {
  using namespace mps;
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  TORCH_CHECK(self.dim() == 2, "fused cross entropy expects 2D logits, got ", self.dim(), "D");
  TORCH_CHECK(self.is_contiguous(), "fused cross entropy expects contiguous logits");
  TORCH_CHECK(self.scalar_type() == kFloat || self.scalar_type() == kHalf || self.scalar_type() == kBFloat16,
              "fused cross entropy supports float32/float16/bfloat16, got ",
              self.scalar_type());

  int64_t B = self.size(0);
  int64_t V = self.size(1);

  auto loss = at::empty({B}, self.options().dtype(kFloat));
  auto lse = at::empty({B}, self.options().dtype(kFloat));
  auto weight_per_row = at::empty({B}, self.options().dtype(kFloat));

  Tensor target_i64 = target.dim() == 0 ? target.view({1}) : target;
  target_i64 = target_i64.to(kLong).contiguous();

  bool has_weight = weight.defined();
  // Per-class weight is consumed in fp32 by the kernel. sum_w = sum_c w[c] is
  // precomputed on-device (length-V reduction, done once) for the weighted
  // label-smoothing term, so no .item() sync is needed.
  Tensor weight_f32;
  Tensor sum_w;
  if (has_weight) {
    weight_f32 = weight.to(kFloat).contiguous();
    sum_w = at::sum(weight_f32).view({1});
  } else {
    weight_f32 = at::empty({1}, self.options().dtype(kFloat));
    sum_w = at::empty({1}, self.options().dtype(kFloat));
  }

  CrossEntropyParams params = {};
  params.vocab_size = static_cast<uint32_t>(V);
  params.batch_size = static_cast<uint32_t>(B);
  params.ignore_index = static_cast<int32_t>(ignore_index);
  params.has_weight = has_weight ? 1u : 0u;
  params.label_smoothing = static_cast<float>(label_smoothing);

  std::string kname = fmt::format("cross_entropy_forward_{}", scalarToMetalTypeString(self));

  if (B == 0) {
    return {loss, lse, weight_per_row};
  }

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      auto pso = ce_lib.getPipelineStateForFunc(kname);
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(computeEncoder,
                  self,
                  target_i64,
                  weight_f32,
                  sum_w,
                  loss,
                  lse,
                  weight_per_row,
                  params,
                  stream->getErrorBuffer());

      uint32_t tg_size = ce_threadgroup_size(pso, V);
      [computeEncoder dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    }
  });

  return {loss, lse, weight_per_row};
}

// Backward: gradient of the per-row loss w.r.t. logits. grad_loss carries the
// host reduction factor already (it is the gradient of the unreduced loss).
Tensor _fused_cross_entropy_loss_2d_backward_mps(const Tensor& grad_loss,
                                                 const Tensor& self,
                                                 const Tensor& target,
                                                 const std::optional<Tensor>& weight_opt,
                                                 const Tensor& lse,
                                                 int64_t ignore_index,
                                                 double label_smoothing) {
  using namespace mps;
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  int64_t B = self.size(0);
  int64_t V = self.size(1);

  Tensor grad_loss_f32 = grad_loss.to(kFloat).contiguous();
  auto grad_input = at::empty_like(self);
  Tensor target_i64 = target.dim() == 0 ? target.view({1}) : target;
  target_i64 = target_i64.to(kLong).contiguous();

  bool has_weight = weight.defined();
  Tensor weight_f32;
  Tensor sum_w;
  if (has_weight) {
    weight_f32 = weight.to(kFloat).contiguous();
    sum_w = at::sum(weight_f32).view({1});
  } else {
    weight_f32 = at::empty({1}, self.options().dtype(kFloat));
    sum_w = at::empty({1}, self.options().dtype(kFloat));
  }

  CrossEntropyParams params = {};
  params.vocab_size = static_cast<uint32_t>(V);
  params.batch_size = static_cast<uint32_t>(B);
  params.ignore_index = static_cast<int32_t>(ignore_index);
  params.has_weight = has_weight ? 1u : 0u;
  params.label_smoothing = static_cast<float>(label_smoothing);

  if (B == 0) {
    return grad_input;
  }

  std::string kname = fmt::format("cross_entropy_backward_{}", scalarToMetalTypeString(self));

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      auto pso = ce_lib.getPipelineStateForFunc(kname);
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(computeEncoder, grad_loss_f32, self, target_i64, weight_f32, sum_w, lse, grad_input, params);

      uint32_t tg_size = ce_threadgroup_size(pso, V);
      [computeEncoder dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    }
  });

  return grad_input;
}

// Meta: shape/dtype inference for fake tensors so torch.compile / export and
// functorch can trace the fused op without a real MPS device.

} // namespace at::native
