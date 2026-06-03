//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/binary_cross_entropy_backward_native.h>
#include <ATen/ops/binary_cross_entropy_native.h>
#include <ATen/ops/huber_loss_backward_native.h>
#include <ATen/ops/huber_loss_native.h>
#include <ATen/ops/mse_loss_backward_native.h>
#include <ATen/ops/mse_loss_native.h>
#include <ATen/ops/nll_loss2d_backward_native.h>
#include <ATen/ops/nll_loss2d_forward_native.h>
#include <ATen/ops/nll_loss_backward_native.h>
#include <ATen/ops/nll_loss_forward_native.h>
#include <ATen/ops/smooth_l1_loss_backward_native.h>
#include <ATen/ops/smooth_l1_loss_native.h>
#endif

#include <ATen/ops/huber_loss.h>
#include <ATen/ops/huber_loss_backward.h>
#include <ATen/ops/mse_loss.h>
#include <ATen/ops/mse_loss_backward.h>
#include <ATen/ops/smooth_l1_loss.h>
#include <ATen/ops/smooth_l1_loss_backward.h>
#include <torch/csrc/autograd/custom_function.h>

namespace at::native {
namespace mps {

// ── Metal kernel library (LossOps.metal) ────────────────────────────────────
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/LossOps_metallib.h>
#endif

struct PointwiseLossParams {
  uint32_t N;
  float scale;
  uint32_t reduction;
};
struct SmoothHuberParams {
  uint32_t N;
  float scale;
  uint32_t reduction;
  float beta;
  uint32_t is_huber;
};
struct BCEParams {
  uint32_t N;
  float scale;
  uint32_t reduction;
  uint32_t has_weight;
};

struct NLLParams {
  uint32_t N, C;
  int32_t ignore_index;
  uint32_t reduction, has_weight;
};

static constexpr uint32_t kLossKernelTgsz = 256;
// Cap reduce threadgroup count: avoids O(N/256) barrier calls at large N
static constexpr uint32_t kMaxReduceTGs = 256u;

static void encode_reduce_partials(id<MTLComputeCommandEncoder> enc,
                                   const Tensor& partial,
                                   const Tensor& loss_out,
                                   uint32_t n_tg) {
  const std::string dt_out = scalarToMetalTypeString(loss_out);
  auto pso = lib.getPipelineStateForFunc("loss_reduce_partials_" + dt_out);
  [enc setComputePipelineState:pso];
  mtl_setArgs(enc, partial, loss_out, n_tg);
  [enc dispatchThreads:MTLSizeMake(kLossKernelTgsz, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
}

static void mse_fwd_metal(const Tensor& input_arg, const Tensor& target_arg, int64_t reduction, const Tensor& output) {
  const Tensor input = input_arg.is_contiguous() ? input_arg : input_arg.contiguous();
  const Tensor target = target_arg.is_contiguous() ? target_arg : target_arg.contiguous();
  const uint32_t N = static_cast<uint32_t>(input.numel());
  const std::string dt = scalarToMetalTypeString(input);
  MPSStream* stream = getCurrentMPSStream();
  if (reduction == Reduction::None) {
    // If output is non-contiguous, write to a contiguous temp and copy
    Tensor out_buf = output.is_contiguous() ? output : at::empty_like(output, MemoryFormat::Contiguous);
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto pso = lib.getPipelineStateForFunc("mse_loss_fwd_none_" + dt);
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, input, target, out_buf, N);
        [enc dispatchThreads:MTLSizeMake((N + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
      }
    });
    stream->synchronize(SyncType::COMMIT);
    if (!output.is_contiguous())
      output.copy_(out_buf);
  } else {
    const float scale = (reduction == Reduction::Mean) ? 1.f / static_cast<float>(N) : 1.f;
    PointwiseLossParams p{N, scale, static_cast<uint32_t>(reduction)};
    const uint32_t n_tg = std::min((N + kLossKernelTgsz - 1) / kLossKernelTgsz, kMaxReduceTGs);
    Tensor partial = at::empty({static_cast<int64_t>(n_tg)}, input.options().dtype(at::kFloat));
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto pso = lib.getPipelineStateForFunc("mse_loss_fwd_reduce_" + dt);
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, input, target, partial, p);
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
        encode_reduce_partials(enc, partial, output, n_tg);
      }
    });
    stream->synchronize(SyncType::COMMIT);
  }
}

static void mse_bwd_metal(const Tensor& grad_output,
                          const Tensor& input,
                          const Tensor& target,
                          int64_t reduction,
                          Tensor& grad_input) {
  const Tensor g = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  const Tensor i = input.is_contiguous() ? input : input.contiguous();
  const Tensor t = target.is_contiguous() ? target : target.contiguous();
  const bool need_gi_copy = !grad_input.is_contiguous();
  Tensor gi = need_gi_copy ? at::empty_like(grad_input, at::MemoryFormat::Contiguous) : grad_input;
  const uint32_t N = static_cast<uint32_t>(i.numel());
  const float scale = (reduction == Reduction::Mean) ? 1.f / static_cast<float>(N) : 1.f;
  PointwiseLossParams p{N, scale, static_cast<uint32_t>(reduction)};
  const std::string dt = scalarToMetalTypeString(i);
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc("mse_loss_bwd_" + dt);
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, g, i, t, gi, p);
      [enc dispatchThreads:MTLSizeMake((N + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
    }
  });
  stream->synchronize(SyncType::COMMIT);
  if (need_gi_copy)
    grad_input.copy_(gi);
}

static Tensor& bce_loss_metal(const Tensor& input,
                              const Tensor& target,
                              const std::optional<Tensor>& weight_opt,
                              int64_t reduction,
                              Tensor& loss,
                              const std::optional<Tensor>& grad_output_opt,
                              const std::string& op_name) {
  TORCH_CHECK(target.is_same_size(input), op_name + ": target and input tensors must have identical shapes");
  const bool is_bwd = grad_output_opt.has_value();
  c10::MaybeOwned<Tensor> wmo = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *wmo;
  loss.resize_((reduction == Reduction::None || is_bwd) ? target.sizes() : IntArrayRef({}));
  TORCH_CHECK(loss.is_mps());
  const uint32_t N = static_cast<uint32_t>(input.numel());
  const float scale = (reduction == Reduction::Mean) ? 1.f / static_cast<float>(N) : 1.f;
  BCEParams p{N, scale, static_cast<uint32_t>(reduction), weight.defined() ? 1u : 0u};
  const std::string dt = scalarToMetalTypeString(input);
  MPSStream* stream = getCurrentMPSStream();
  // Expand weight to input shape: Metal kernel reads weight[i] linearly for i in [0, N)
  const Tensor wt = weight.defined()
      ? (weight.sizes() == input.sizes() ? (weight.is_contiguous() ? weight : weight.contiguous())
                                         : weight.expand_as(input).contiguous())
      : input;
  if (is_bwd) {
    // grad_out may have stride-0 layout (e.g. from sum().backward() → expand)
    const Tensor grad_out_c =
        grad_output_opt.value().is_contiguous() ? grad_output_opt.value() : grad_output_opt.value().contiguous();
    const Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    const Tensor target_c = target.is_contiguous() ? target : target.contiguous();
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto pso = lib.getPipelineStateForFunc("bce_loss_bwd_" + dt);
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, grad_out_c, input_c, target_c, wt, loss, p);
        [enc dispatchThreads:MTLSizeMake((N + 15u) / 16u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
      }
    });
    stream->synchronize(SyncType::COMMIT);
    return loss;
  }
  // Forward path: ensure input/target are contiguous before passing to kernel.
  const Tensor input_c = input.is_contiguous() ? input : input.contiguous();
  const Tensor target_c = target.is_contiguous() ? target : target.contiguous();
  const bool need_loss_copy = !loss.is_contiguous();
  Tensor loss_buf = need_loss_copy ? at::empty_like(loss, at::MemoryFormat::Contiguous) : loss;
  if (reduction == Reduction::None) {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto pso = lib.getPipelineStateForFunc("bce_loss_fwd_none_" + dt);
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, input_c, target_c, wt, loss_buf, p);
        [enc dispatchThreads:MTLSizeMake((N + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
      }
    });
    stream->synchronize(SyncType::COMMIT);
  } else {
    const uint32_t n_tg = std::min((N + kLossKernelTgsz - 1) / kLossKernelTgsz, kMaxReduceTGs);
    Tensor partial = at::empty({static_cast<int64_t>(n_tg)}, input.options().dtype(at::kFloat));
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto pso = lib.getPipelineStateForFunc("bce_loss_fwd_reduce_" + dt);
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, input_c, target_c, wt, partial, p);
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
        encode_reduce_partials(enc, partial, loss, n_tg);
      }
    });
    stream->synchronize(SyncType::COMMIT);
  }
  if (need_loss_copy)
    loss.copy_(loss_buf);
  return loss;
}

static void smooth_huber_fwd_metal(const Tensor& input,
                                   const Tensor& target,
                                   int64_t reduction,
                                   float beta,
                                   uint32_t is_huber,
                                   const Tensor& output) {
  // Metal kernels are only instantiated for float/half/bfloat. Promote
  // integer / bool inputs to float (mirrors MPSGraph baseline).
  if (c10::isIntegralType(input.scalar_type(), /*includeBool=*/true)) {
    Tensor output_f = at::empty_like(output, output.options().dtype(at::kFloat));
    smooth_huber_fwd_metal(input.to(at::kFloat), target.to(at::kFloat), reduction, beta, is_huber, output_f);
    output.copy_(output_f);
    return;
  }
  if (input.numel() == 0 || target.numel() == 0) {
    reduction == Reduction::Mean ? output.fill_(std::numeric_limits<float>::quiet_NaN()) : output.zero_();
    return;
  }
  const Tensor input_c = input.is_contiguous() ? input : input.contiguous();
  const Tensor target_c = target.is_contiguous() ? target : target.contiguous();
  const bool need_out_copy = !output.is_contiguous();
  Tensor out_buf = need_out_copy ? at::empty_like(output, at::MemoryFormat::Contiguous) : output;
  const uint32_t N = static_cast<uint32_t>(input_c.numel());
  const float scale = (reduction == Reduction::Mean) ? 1.f / static_cast<float>(N) : 1.f;
  SmoothHuberParams p{N, scale, static_cast<uint32_t>(reduction), beta, is_huber};
  const std::string dt = scalarToMetalTypeString(input_c);
  MPSStream* stream = getCurrentMPSStream();
  if (reduction == Reduction::None) {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto pso = lib.getPipelineStateForFunc("smooth_huber_fwd_none_" + dt);
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, input_c, target_c, out_buf, p);
        [enc dispatchThreads:MTLSizeMake((N + 3u) / 4u, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
      }
    });
    stream->synchronize(SyncType::COMMIT);
  } else {
    const uint32_t n_tg = std::min((N + kLossKernelTgsz - 1) / kLossKernelTgsz, kMaxReduceTGs);
    Tensor partial = at::empty({static_cast<int64_t>(n_tg)}, input.options().dtype(at::kFloat));
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto pso = lib.getPipelineStateForFunc("smooth_huber_fwd_reduce_" + dt);
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, input_c, target_c, partial, p);
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
        encode_reduce_partials(enc, partial, out_buf, n_tg);
      }
    });
    stream->synchronize(SyncType::COMMIT);
  }
  if (need_out_copy)
    output.copy_(out_buf);
}

static void smooth_huber_bwd_metal(const Tensor& grad_output,
                                   const Tensor& input,
                                   const Tensor& target,
                                   int64_t reduction,
                                   float beta,
                                   uint32_t is_huber,
                                   Tensor& grad_input) {
  if (grad_input.numel() == 0)
    return;
  if (c10::isIntegralType(input.scalar_type(), /*includeBool=*/true)) {
    Tensor gi_f = at::empty_like(grad_input, grad_input.options().dtype(at::kFloat));
    smooth_huber_bwd_metal(grad_output.to(at::kFloat), input.to(at::kFloat),
                           target.to(at::kFloat), reduction, beta, is_huber, gi_f);
    grad_input.copy_(gi_f);
    return;
  }
  const Tensor g = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  const Tensor i = input.is_contiguous() ? input : input.contiguous();
  const Tensor t = target.is_contiguous() ? target : target.contiguous();
  const bool need_gi_copy = !grad_input.is_contiguous();
  Tensor gi = need_gi_copy ? at::empty_like(grad_input, at::MemoryFormat::Contiguous) : grad_input;
  const uint32_t N = static_cast<uint32_t>(i.numel());
  const float scale = (reduction == Reduction::Mean) ? 1.f / static_cast<float>(N) : 1.f;
  SmoothHuberParams p{N, scale, static_cast<uint32_t>(reduction), beta, is_huber};
  const std::string dt = scalarToMetalTypeString(i);
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc("smooth_huber_bwd_" + dt);
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, g, i, t, gi, p);
      [enc dispatchThreads:MTLSizeMake((N + 15u) / 16u, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
    }
  });
  stream->synchronize(SyncType::COMMIT);
  if (need_gi_copy)
    grad_input.copy_(gi);
}

static std::string reductionToString(int64_t reduction) {
  switch (reduction) {
    case Reduction::Mean:
      return "Mean";
    case Reduction::Sum:
      return "Sum";
    default:
      return "None";
  }
}

static MPSGraphTensor* reduceTensor(MPSGraphTensor* tensor,
                                    int64_t reduction,
                                    MPSGraph* mpsGraph,
                                    NSUInteger axesCount) {
  NSMutableArray<NSNumber*>* axes = [NSMutableArray<NSNumber*> arrayWithCapacity:axesCount];
  for (NSUInteger i = 0; i < axesCount; i++)
    axes[i] = @(i);

  switch (reduction) {
    case Reduction::Mean:
      return [mpsGraph meanOfTensor:tensor axes:axes name:@"reductionMeanTensor"];
    case Reduction::Sum:
      return [mpsGraph reductionSumWithTensor:tensor axes:axes name:@"reductionSumTensor"];
    default:
      assert(reduction == Reduction::None);
      return tensor;
  }
}

static Tensor& mse_loss_backward_out_impl(const Tensor& grad_output,
                                          const Tensor& input,
                                          const Tensor& target,
                                          int64_t reduction,
                                          Tensor& grad_input,
                                          const std::string& op_name) {
  TORCH_CHECK(target.is_same_size(input), op_name + ": target and input tensors must have identical shapes")

  if ((input.numel() == 0) || (target.numel() == 0) || (grad_output.numel() == 0)) {
    reduction == Reduction::Mean ? grad_input.fill_(std::numeric_limits<float>::quiet_NaN()) : grad_input.zero_();
    return grad_input;
  }

  mse_bwd_metal(grad_output, input, target, reduction, grad_input);
  return grad_input;
}

// BCELoss: Metal dispatch; see bce_loss_metal() above.

static inline MPSGraphTensor* divisionNoNaN(MPSGraph* mpsGraph, MPSGraphTensor* divident, MPSGraphTensor* divisor) {
  auto* div = [mpsGraph divisionWithPrimaryTensor:divident
                                  secondaryTensor:castMPSTensor(mpsGraph, divisor, divident.dataType)
                                             name:@"divisionTensor"];
  // Replace NaNs with 0 for divident elements equal to 0
  return [mpsGraph selectWithPredicateTensor:castMPSTensor(mpsGraph, divisor, MPSDataTypeBool)
                         truePredicateTensor:div
                        falsePredicateTensor:[mpsGraph constantWithScalar:0.0 dataType:div.dataType]
                                        name:nil];
}

// NLLLoss
static void nll_loss_fwd_metal(Tensor& output,
                               Tensor& total_weight,
                               const Tensor& input_arg,
                               const Tensor& target_arg,
                               const Tensor& weight,
                               int64_t reduction,
                               int64_t ignore_index) {
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(output.scalar_type()),
                              "nll_loss for complex is not supported for MPS");
  if (weight.defined()) {
    TORCH_CHECK(input_arg.scalar_type() == weight.scalar_type(),
                "expected scalar type ",
                input_arg.scalar_type(),
                " but found ",
                weight.scalar_type());
  }
  if (c10::isIntegralType(input_arg.scalar_type(), /*includeBool=*/true)) {
    Tensor input_f = input_arg.to(at::kFloat);
    Tensor weight_f = weight.defined() ? weight.to(at::kFloat) : weight;
    Tensor output_f = at::empty_like(output, output.options().dtype(at::kFloat));
    Tensor tw_f = at::empty_like(total_weight, total_weight.options().dtype(at::kFloat));
    nll_loss_fwd_metal(output_f, tw_f, input_f, target_arg, weight_f, reduction, ignore_index);
    output.copy_(output_f);
    total_weight.copy_(tw_f);
    return;
  }
  auto input = input_arg.dim() == 1 ? input_arg.view({1, input_arg.size(0)}) : input_arg;
  auto target = target_arg.dim() == 0 ? target_arg.view({1}) : target_arg;

  const uint32_t N = static_cast<uint32_t>(target.numel());
  const uint32_t C = static_cast<uint32_t>(input.size(1));

  if (reduction == Reduction::None)
    output.resize_(target_arg.sizes());
  else
    output.resize_({});
  total_weight.resize_({});

  // Kernels are stride-blind: force contiguous copies before dispatch.
  input = input.contiguous();
  const Tensor weight_c = weight.defined() ? (weight.is_contiguous() ? weight : weight.contiguous()) : weight;
  const bool need_out_copy = !output.is_contiguous();
  Tensor out_buf = need_out_copy ? at::empty_like(output, at::MemoryFormat::Contiguous) : output;

  if (N == 0 || input.numel() == 0) {
    reduction == Reduction::Mean ? output.fill_(std::numeric_limits<float>::quiet_NaN()) : output.zero_();
    total_weight.zero_();
    return;
  }

  NLLParams p{N, C, static_cast<int32_t>(ignore_index), static_cast<uint32_t>(reduction), weight.defined() ? 1u : 0u};
  const std::string dt = scalarToMetalTypeString(input);
  // Metal NLL kernels read target as int32; cast if needed (target is usually int64)
  const Tensor tgt_i32 = target.scalar_type() == at::kInt ? target : target.to(at::kInt);
  // Use input as a valid dummy buffer when no class weights provided
  const Tensor& wt = weight_c.defined() ? weight_c : input;
  MPSStream* stream = getCurrentMPSStream();

  if (reduction == Reduction::None) {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto pso = lib.getPipelineStateForFunc("nll_loss_fwd_none_" + dt);
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, input, tgt_i32, wt, out_buf, p, stream->getErrorBuffer());
        [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
      }
    });
    stream->synchronize(SyncType::COMMIT);
    total_weight.fill_(static_cast<float>(N));
  } else {
    const uint32_t n_tg = std::min((N + kLossKernelTgsz - 1) / kLossKernelTgsz, kMaxReduceTGs);
    Tensor partial = at::empty({static_cast<int64_t>(n_tg)}, input.options().dtype(at::kFloat));
    Tensor wpartial = at::empty({static_cast<int64_t>(n_tg)}, input.options().dtype(at::kFloat));
    // Weight sum always accumulated as float; copy back to total_weight (any dtype) after commit
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        // Phase 1: per-threadgroup loss and weight sums
        auto pso = lib.getPipelineStateForFunc("nll_loss_fwd_reduce_" + dt);
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, input, tgt_i32, wt, partial, wpartial, p, stream->getErrorBuffer());
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
        // Phase 2a: reduce loss partials → typed output[0]
        encode_reduce_partials(enc, partial, out_buf, n_tg);
        // Phase 2b: reduce weight partials → typed total_weight[0]
        encode_reduce_partials(enc, wpartial, total_weight, n_tg);
        // Phase 3 (Mean only): out_buf[0] /= total_weight[0]
        if (reduction == Reduction::Mean) {
          auto pso3 = lib.getPipelineStateForFunc("nll_finalize_mean_" + dt);
          [enc setComputePipelineState:pso3];
          mtl_setArgs(enc, out_buf, total_weight);
          [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        }
      }
    });
    stream->synchronize(SyncType::COMMIT);
  }
  if (need_out_copy)
    output.copy_(out_buf);
}

static void nll_loss_bwd_metal(Tensor& grad_input,
                               const Tensor& grad_output,
                               const Tensor& input_arg,
                               const Tensor& target_arg,
                               const Tensor& weight,
                               int64_t reduction,
                               int64_t ignore_index,
                               const Tensor& total_weight) {
  if (c10::isIntegralType(input_arg.scalar_type(), /*includeBool=*/true)) {
    Tensor input_f = input_arg.to(at::kFloat);
    Tensor weight_f = weight.defined() ? weight.to(at::kFloat) : weight;
    Tensor gi_f = at::empty_like(grad_input, grad_input.options().dtype(at::kFloat));
    nll_loss_bwd_metal(gi_f, grad_output.to(at::kFloat), input_f, target_arg, weight_f,
                       reduction, ignore_index, total_weight.to(at::kFloat));
    grad_input.copy_(gi_f);
    return;
  }
  auto input = input_arg.dim() == 1 ? input_arg.view({1, input_arg.size(0)}) : input_arg;
  auto target = target_arg.dim() == 0 ? target_arg.view({1}) : target_arg;

  const uint32_t N = static_cast<uint32_t>(target.numel());
  const uint32_t C = static_cast<uint32_t>(input.size(1));

  grad_input.resize_as_(input_arg);
  grad_input.zero_();

  if (N == 0 || input.numel() == 0)
    return;

  // Kernel is stride-blind: force contiguous copies for input/weight,
  // and route writes through a contiguous grad_input buffer if needed.
  input = input.contiguous();
  const Tensor weight_c = weight.defined() ? (weight.is_contiguous() ? weight : weight.contiguous()) : weight;
  const bool need_gi_copy = !grad_input.is_contiguous();
  Tensor gi = need_gi_copy ? at::empty_like(grad_input, at::MemoryFormat::Contiguous) : grad_input;

  NLLParams p{N, C, static_cast<int32_t>(ignore_index), static_cast<uint32_t>(reduction), weight_c.defined() ? 1u : 0u};
  const std::string dt = scalarToMetalTypeString(input);
  const Tensor tgt_i32 = target.scalar_type() == at::kInt ? target : target.to(at::kInt);
  const Tensor& wt = weight_c.defined() ? weight_c : input;
  // grad_output may have stride-0 (expanded) layout when coming from sum().backward()
  const Tensor grad_out_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  MPSStream* stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc("nll_loss_bwd_" + dt);
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, grad_out_c, tgt_i32, wt, gi, total_weight, p, stream->getErrorBuffer());
      [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
    }
  });
  stream->synchronize(SyncType::COMMIT);
  if (need_gi_copy)
    grad_input.copy_(gi);
}

static void nllnd_loss_backward_impl(Tensor& grad_input_arg,
                                     const Tensor& grad_output_arg,
                                     const Tensor& input_arg,
                                     const Tensor& target_arg,
                                     const Tensor& weight_arg,
                                     int64_t reduction,
                                     int64_t ignore_index,
                                     const Tensor& total_weight,
                                     bool is2D) {
  if (grad_input_arg.numel() == 0) {
    return;
  }
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* targetTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* totalWeightTensor_ = nil;
    MPSGraphTensor* gradInputTensor_ = nil;
    MPSGraphTensor* gradOutputTensor_ = nil;
  };
  bool isWeightsArrayValid = weight_arg.defined() && weight_arg.numel() > 0;
  bool isTargetCasted = target_arg.scalar_type() != ScalarType::Long;
  int64_t channel_dim = grad_input_arg.dim() < 2 ? 0 : 1;
  auto input = input_arg.dim() == 1 ? input_arg.view({1, input_arg.size(0)}) : input_arg;
  auto target = target_arg.dim() == 0 ? target_arg.view({1}) : target_arg;
  auto grad_input = grad_input_arg.dim() == 1 ? grad_input_arg.view({1, grad_input_arg.size(0)}) : grad_input_arg;
  auto numClasses = grad_input.sizes()[1];
  auto weight = weight_arg;
  auto grad_output = grad_output_arg;

  if (isWeightsArrayValid) {
    std::vector<int64_t> weightShape(input.dim(), 1);
    weightShape[1] = input.size(1);
    weight = weight_arg.view(weightShape);
  }
  if (grad_output_arg.dim() < grad_input.dim() && grad_output_arg.dim() > 0) {
    grad_output = grad_output_arg.unsqueeze(channel_dim);
  }
  @autoreleasepool {
    std::string key = "nllnd_loss_backward" + getTensorsStringKey({input, grad_output, target, weight, total_weight}) +
        std::to_string(numClasses) + ":" + std::to_string(ignore_index) + ":" + std::to_string(isWeightsArrayValid) +
        ":" + std::to_string(isTargetCasted) + ":" + reductionToString(reduction);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
      MPSGraphTensor* targetTensor = mpsGraphRankedPlaceHolder(mpsGraph, target);
      MPSGraphTensor* castedTargetTensor =
          isTargetCasted ? castMPSTensor(mpsGraph, targetTensor, MPSDataTypeInt64) : targetTensor;
      MPSGraphTensor* weightTensor = nil;
      if (isWeightsArrayValid) {
        weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight);
      }
      MPSGraphTensor* totalWeightTensor = mpsGraphRankedPlaceHolder(mpsGraph, total_weight);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);

      MPSGraphTensor* updatedTargetTensor = castedTargetTensor;

      // Replace ignored_index with length depth + 1 so that oneHotAPI ignores it
      MPSGraphTensor* ignoreIndexTensor = [mpsGraph constantWithScalar:ignore_index dataType:MPSDataTypeInt64];
      MPSGraphTensor* numClassesTensor = [mpsGraph constantWithScalar:(numClasses + 1) dataType:MPSDataTypeInt64];
      MPSGraphTensor* isEqualTensor = [mpsGraph equalWithPrimaryTensor:castedTargetTensor
                                                       secondaryTensor:ignoreIndexTensor
                                                                  name:@"isEqualTensor"];
      updatedTargetTensor = [mpsGraph selectWithPredicateTensor:isEqualTensor
                                            truePredicateTensor:numClassesTensor
                                           falsePredicateTensor:castedTargetTensor
                                                           name:@"predicateTensor"];

      // oneHotWithIndicesTensor only works for Float32 dtype
      // cast it explicitly later if needed
      auto* oneHotTensor = [mpsGraph oneHotWithIndicesTensor:updatedTargetTensor
                                                       depth:numClasses
                                                        axis:1
                                                    dataType:MPSDataTypeFloat32
                                                     onValue:-1.0f
                                                    offValue:0.0f
                                                        name:nil];
      oneHotTensor = castMPSTensor(mpsGraph, oneHotTensor, [inputTensor dataType]);
      if (isWeightsArrayValid) {
        oneHotTensor = [mpsGraph multiplicationWithPrimaryTensor:oneHotTensor
                                                 secondaryTensor:weightTensor
                                                            name:@"scaleByWeightTensor"];
      }
      if (reduction == Reduction::Mean) {
        oneHotTensor = divisionNoNaN(mpsGraph, oneHotTensor, totalWeightTensor);
      }
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:oneHotTensor
                                                                  secondaryTensor:gradOutputTensor
                                                                             name:nil];
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->targetTensor_ = targetTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->totalWeightTensor_ = totalWeightTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input);
    auto gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    auto targetPlaceholder = Placeholder(cachedGraph->targetTensor_, target);
    Placeholder weightPlaceholder = Placeholder();
    if (isWeightsArrayValid) {
      weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight);
    }
    auto totalWeightPlaceholder = Placeholder(cachedGraph->totalWeightTensor_, total_weight);
    auto gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[targetPlaceholder.getMPSGraphTensor()] = targetPlaceholder.getMPSGraphTensorData();
    feeds[totalWeightPlaceholder.getMPSGraphTensor()] = totalWeightPlaceholder.getMPSGraphTensorData();
    feeds[gradOutputPlaceholder.getMPSGraphTensor()] = gradOutputPlaceholder.getMPSGraphTensorData();

    if (isWeightsArrayValid) {
      feeds[weightPlaceholder.getMPSGraphTensor()] = weightPlaceholder.getMPSGraphTensorData();
    }
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
}

static void nllnd_loss_forward_impl(Tensor& output,
                                    Tensor& total_weight,
                                    const Tensor& input_arg,
                                    const Tensor& target_arg,
                                    const Tensor& weight,
                                    int64_t reduction,
                                    int64_t ignore_index,
                                    bool is2D) {
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(output.scalar_type()),
                              "nlld_loss for complex is not supported for MPS");
  if (weight.defined()) {
    TORCH_CHECK(input_arg.scalar_type() == weight.scalar_type(),
                "expected scalar type ",
                input_arg.scalar_type(),
                " but found ",
                weight.scalar_type());
  }
  std::vector<long long> reshapedTarget(target_arg.sizes().begin(), target_arg.sizes().end());
  reshapedTarget.push_back(1);

  Tensor batchSizeTensor = at::empty_like(input_arg).resize_(IntArrayRef(1));
  float batchVal = 1.0f;
  for (size_t i = 0; i < reshapedTarget.size(); ++i)
    batchVal *= reshapedTarget[i];
  batchSizeTensor[0] = batchVal;

  if (reduction == Reduction::None)
    output.resize_(target_arg.sizes());
  if (reduction == Reduction::Sum)
    output.resize_({});
  if (reduction == Reduction::Mean)
    output.resize_({});

  TORCH_CHECK(output.is_mps());

  // Empty output
  if (output.numel() == 0)
    return;

  // https://github.com/pytorch/pytorch/blob/042f2f7746a064f1527d95d1f1d712b4f0b34186/aten/src/ATen/native/cuda/Loss.cu#L335-L346
  if (target_arg.numel() == 0) {
    // Here target (and input) have zero elements
    // Mean reduction on empty tensors produces NaN. See the discussion in
    // https://github.com/pytorch/pytorch/pull/64572#issuecomment-926504162
    if (reduction == Reduction::Mean) {
      output.fill_(std::numeric_limits<double>::quiet_NaN());
    } else {
      output.zero_();
    }
    total_weight.zero_();
    return;
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* targetTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* batchSizeTensor_ = nil;
    MPSGraphTensor* totalWeightTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  auto input = input_arg.dim() == 1 ? input_arg.view({1, input_arg.size(0)}) : input_arg;
  auto target = target_arg.dim() == 0 ? target_arg.view({1}) : target_arg;

  @autoreleasepool {
    bool isWeightsArrayValid = (weight.numel() > 0);
    bool isTargetCasted = target.scalar_type() != ScalarType::Long;

    MPSShape* input_shape = getMPSShape(input);
    MPSShape* target_shape = getMPSShape(target);
    MPSShape* weight_shape = getMPSShape(weight);

    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];

    // TODO: Make the key
    std::string key = "nllnd_loss_forward_impl:" + std::to_string(ignore_index) + ":" +
        std::to_string(isWeightsArrayValid) + ":" + reductionToString(reduction) + ":" + [ns_shape_key UTF8String] +
        ":" + getMPSTypeString(input) + ":" + getMPSTypeString(target) + ":" + std::to_string(isTargetCasted) + ":" +
        getMPSTypeString(weight);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), input_shape);
      MPSGraphTensor* targetTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(target), target_shape);
      MPSGraphTensor* castedTargetTensor =
          isTargetCasted ? castMPSTensor(mpsGraph, targetTensor, MPSDataTypeInt64) : targetTensor;
      MPSGraphTensor* weightTensor = nil;
      if (isWeightsArrayValid)
        weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(weight), weight_shape);
      MPSGraphTensor* mps_batchSizeTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(batchSizeTensor));

      MPSGraphTensor* mpsGraphBatchSizeTensor = mps_batchSizeTensor;

      // The transposes are needed to get the class dimension (dim 1) to the inner most dim for gather op.
      // The transpose become nop in the 2D case.
      MPSGraphTensor* mpsTransposeTensor = inputTensor;
      int classDim = 1;
      int lastDim = input.sizes().size() - 1;
      mpsTransposeTensor = [mpsGraph transposeTensor:inputTensor dimension:classDim withDimension:lastDim name:nil];
      for (int i = 0; i < lastDim - 2; ++i) {
        mpsTransposeTensor = [mpsGraph transposeTensor:mpsTransposeTensor
                                             dimension:classDim + i
                                         withDimension:classDim + i + 1
                                                  name:nil];
      }

      MPSGraphTensor* mpsGatherTensor = [mpsGraph gatherWithUpdatesTensor:mpsTransposeTensor
                                                            indicesTensor:castedTargetTensor
                                                                     axis:lastDim
                                                          batchDimensions:lastDim
                                                                     name:@"gatherTensor"];

      MPSGraphTensor* mpsGraphZeroTensor = [mpsGraph constantWithScalar:0.0 dataType:mpsGatherTensor.dataType];
      MPSGraphTensor* mpsGraphOneTensor = [mpsGraph constantWithScalar:1.0 dataType:mpsGatherTensor.dataType];
      MPSGraphTensor* mpsGraphIndexTensor = [mpsGraph constantWithScalar:ignore_index dataType:MPSDataTypeInt64];
      MPSGraphTensor* mpsGraphIsEqualTensor = [mpsGraph equalWithPrimaryTensor:castedTargetTensor
                                                               secondaryTensor:mpsGraphIndexTensor
                                                                          name:@"isEqualTensor"];
      // Zero out loss
      mpsGatherTensor = [mpsGraph selectWithPredicateTensor:mpsGraphIsEqualTensor
                                        truePredicateTensor:mpsGraphZeroTensor
                                       falsePredicateTensor:mpsGatherTensor
                                                       name:@"predicateTensor"];

      if (isWeightsArrayValid) {
        MPSGraphTensor* weightGatherTensor = [mpsGraph gatherWithUpdatesTensor:weightTensor
                                                                 indicesTensor:castedTargetTensor
                                                                          axis:0
                                                               batchDimensions:0
                                                                          name:@"weightGatherTensor"];
        mpsGatherTensor = [mpsGraph multiplicationWithPrimaryTensor:weightGatherTensor
                                                    secondaryTensor:mpsGatherTensor
                                                               name:@"scaledLossTensor"];
        mpsGraphOneTensor = weightGatherTensor;
      }

      // Compute new batch size
      MPSGraphTensor* mpsSelectOneTensor = [mpsGraph selectWithPredicateTensor:mpsGraphIsEqualTensor
                                                           truePredicateTensor:mpsGraphZeroTensor
                                                          falsePredicateTensor:mpsGraphOneTensor
                                                                          name:@"predicateOneTensor"];

      MPSGraphTensor* mpsGraphNegTensor = [mpsGraph negativeWithTensor:mpsGatherTensor name:@"negativeTensor"];

      MPSGraphTensor* mpsGraphReducedTensor = mpsGraphNegTensor;

      if (!(reduction == Reduction::None)) {
        mpsGraphReducedTensor = [mpsGraph reductionSumWithTensor:mpsGraphNegTensor axes:nil name:@"reductionSumTensor"];
        if (reduction == Reduction::Mean) {
          mpsGraphBatchSizeTensor = [mpsGraph reductionSumWithTensor:mpsSelectOneTensor
                                                                axes:nil
                                                                name:@"batchSizeReductionTensor"];
          mpsGraphReducedTensor = [mpsGraph divisionWithPrimaryTensor:mpsGraphReducedTensor
                                                      secondaryTensor:mpsGraphBatchSizeTensor
                                                                 name:@"divisionTensor"];
        }
      }

      mpsGraphReducedTensor = [mpsGraph reshapeTensor:mpsGraphReducedTensor withShape:getMPSShape(output) name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->targetTensor_ = targetTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->batchSizeTensor_ = mps_batchSizeTensor;
      newCachedGraph->totalWeightTensor_ = mpsGraphBatchSizeTensor;
      newCachedGraph->outputTensor_ = mpsGraphReducedTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, input, nil, true, MPSDataTypeInvalid, false);
    Placeholder targetPlaceholder =
        Placeholder(cachedGraph->targetTensor_, target, nil, true, MPSDataTypeInvalid, false);
    Placeholder weightPlaceholder = Placeholder();
    if (isWeightsArrayValid)
      weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight, nil, true, MPSDataTypeInvalid, false);
    Placeholder batchSizePlaceholder =
        Placeholder(cachedGraph->batchSizeTensor_, batchSizeTensor, nil, true, MPSDataTypeInvalid, false);
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor_, output, nil, true, MPSDataTypeInvalid, false);
    Placeholder totalWeightsPlaceholder =
        Placeholder(cachedGraph->totalWeightTensor_, total_weight, nil, true, MPSDataTypeInvalid, false);

    // Create dictionary of inputs and outputs
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
        [[[NSMutableDictionary alloc] initWithCapacity:4] autorelease];
    feeds[selfPlaceholder.getMPSGraphTensor()] = selfPlaceholder.getMPSGraphTensorData();
    feeds[targetPlaceholder.getMPSGraphTensor()] = targetPlaceholder.getMPSGraphTensorData();
    feeds[batchSizePlaceholder.getMPSGraphTensor()] = batchSizePlaceholder.getMPSGraphTensorData();

    if (isWeightsArrayValid)
      feeds[weightPlaceholder.getMPSGraphTensor()] = weightPlaceholder.getMPSGraphTensorData();

    auto results = dictionaryFromPlaceholders(outputPlaceholder, totalWeightsPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return;
}

static void smooth_l1_loss_impl(const Tensor& input,
                                const Tensor& target,
                                const int64_t reduction,
                                double beta,
                                const Tensor& output,
                                MPSShape* /*mpsInputShape*/,
                                MPSShape* /*mpsOutputShape*/) {
  smooth_huber_fwd_metal(input, target, reduction, static_cast<float>(beta), 0u, output);
}

static void smooth_l1_loss_template(const Tensor& input,
                                    const Tensor& target,
                                    const int64_t reduction,
                                    double beta,
                                    const Tensor& output) {
  TORCH_CHECK(beta >= 0, "smooth_l1_loss does not support negative values for beta.");
  TORCH_CHECK(input.is_mps());
  TORCH_CHECK(target.is_mps());
  TORCH_CHECK_NOT_IMPLEMENTED(input.scalar_type() != kLong, "MPS doesn't know how to do square_i64");
  if ((input.numel() == 0) || (target.numel() == 0)) {
    reduction == Reduction::Mean ? output.fill_(std::numeric_limits<float>::quiet_NaN()) : output.zero_();
    return;
  }
  MPSShape* mpsInputShape = nil;
  MPSShape* mpsOutputShape = nil;

  // Determine the shape of the output
  // If the reduction is 'mean' or 'sum', the output shape is a scalar,
  // otherwise, the output shape is the same shape as input
  if (reduction == Reduction::Mean || reduction == Reduction::Sum) {
    // Output: scalar, if reduction is 'mean' or 'sum'
    IntArrayRef input_shape = input.sizes();
    int64_t num_input_dims = input_shape.size();
    NSMutableArray<NSNumber*>* apparent_input_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    int64_t num_in_elements = 1;
    for (int i = 0; i < num_input_dims; i++) {
      num_in_elements *= input_shape[i];
    }
    apparent_input_shape[0] = [NSNumber numberWithInt:num_in_elements];

    // Output is a single value in case reduction is set to mean or sum
    NSMutableArray<NSNumber*>* apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    apparent_out_shape[0] = @1;
    mpsInputShape = apparent_input_shape;
    mpsOutputShape = apparent_out_shape;
  } else {
    // Output: If reduction is 'none', then (N, *); same shape as the input
    assert(reduction == Reduction::None);
    mpsInputShape = getMPSShape(input);
    mpsOutputShape = mpsInputShape;
    // resize_tensor(&output);
  }
  TORCH_CHECK(output.is_mps());

  smooth_l1_loss_impl(input, target, reduction, beta, output, mpsInputShape, mpsOutputShape);
}

static void smooth_l1_loss_backward_impl(const Tensor& grad_output,
                                         const Tensor& input,
                                         const Tensor& target,
                                         int64_t reduction,
                                         double beta,
                                         Tensor& grad_input) {
  smooth_huber_bwd_metal(grad_output, input, target, reduction, static_cast<float>(beta), 0u, grad_input);
}

// =============================================================================
// Fused fwd (writes loss + saved_dg) for Huber / SmoothL1 / MSE.
// Used by AutogradMPS overrides to enable a single-kernel-launch backward.
// =============================================================================
static std::tuple<Tensor, Tensor> huber_or_sl1_fwd_metal_sg(const Tensor& input,
                                                            const Tensor& target,
                                                            int64_t reduction,
                                                            float beta,
                                                            uint32_t is_huber) {
  Tensor loss = at::empty_like(input); // only called for reduction=None
  Tensor saved_dg = at::empty_like(input);
  if (input.numel() == 0)
    return {loss, saved_dg};
  const uint32_t N = static_cast<uint32_t>(input.numel());
  const float scale = 1.f;
  SmoothHuberParams p{N, scale, static_cast<uint32_t>(reduction), beta, is_huber};
  const std::string dt = scalarToMetalTypeString(input);
  const std::string fn_name = (is_huber ? "huber_fwd_sg_" : "smooth_l1_fwd_sg_") + dt;
  Tensor input_c = input.is_contiguous() ? input : input.contiguous();
  Tensor target_c = target.is_contiguous() ? target : target.contiguous();
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(fn_name);
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, input_c, target_c, loss, saved_dg, p);
      const uint32_t vec_n = 8u;
      [enc dispatchThreads:MTLSizeMake((N + vec_n - 1u) / vec_n, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
    }
  });
  stream->synchronize(SyncType::COMMIT);
  return {loss, saved_dg};
}

static Tensor huber_or_sl1_bwd_metal_sg(const Tensor& grad_output,
                                        const Tensor& saved_dg,
                                        int64_t reduction,
                                        float beta,
                                        uint32_t is_huber) {
  Tensor grad_in = at::empty_like(saved_dg);
  const uint32_t N = static_cast<uint32_t>(saved_dg.numel());
  const float scale = (reduction == Reduction::Mean) ? 1.f / static_cast<float>(N) : 1.f;
  SmoothHuberParams p{N, scale, static_cast<uint32_t>(reduction), beta, is_huber};
  const std::string dt = scalarToMetalTypeString(saved_dg);
  const Tensor grad_out_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc("huber_or_sl1_bwd_sg_" + dt);
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, grad_out_c, saved_dg, grad_in, p);
      const uint32_t vec_n = 8u;
      [enc dispatchThreads:MTLSizeMake((N + vec_n - 1u) / vec_n, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
    }
  });
  stream->synchronize(SyncType::COMMIT);
  return grad_in;
}

static std::tuple<Tensor, Tensor> mse_fwd_metal_sg(const Tensor& input, const Tensor& target) {
  Tensor loss = at::empty_like(input);
  Tensor saved_dg = at::empty_like(input);
  if (input.numel() == 0)
    return {loss, saved_dg};
  const uint32_t N = static_cast<uint32_t>(input.numel());
  const std::string dt = scalarToMetalTypeString(input);
  const Tensor input_c = input.is_contiguous() ? input : input.contiguous();
  const Tensor target_c = target.is_contiguous() ? target : target.contiguous();
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc("mse_fwd_sg_" + dt);
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, input_c, target_c, loss, saved_dg, N);
      const uint32_t vec_n = 8u;
      [enc dispatchThreads:MTLSizeMake((N + vec_n - 1u) / vec_n, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
    }
  });
  stream->synchronize(SyncType::COMMIT);
  return {loss, saved_dg};
}

static Tensor mse_bwd_metal_sg(const Tensor& grad_output, const Tensor& saved_dg, int64_t reduction) {
  Tensor grad_in = at::empty_like(saved_dg);
  const uint32_t N = static_cast<uint32_t>(saved_dg.numel());
  const float scale = (reduction == Reduction::Mean) ? 1.f / static_cast<float>(N) : 1.f;
  PointwiseLossParams p{N, scale, static_cast<uint32_t>(reduction)};
  const std::string dt = scalarToMetalTypeString(saved_dg);
  const Tensor grad_out_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc("mse_bwd_sg_" + dt);
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, grad_out_c, saved_dg, grad_in, p);
      const uint32_t vec_n = 8u;
      [enc dispatchThreads:MTLSizeMake((N + vec_n - 1u) / vec_n, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kLossKernelTgsz, 1, 1)];
    }
  });
  stream->synchronize(SyncType::COMMIT);
  return grad_in;
}

} // namespace mps

// APIs exposed to at::native scope

// HuberLoss

Tensor& huber_loss_out_mps(const Tensor& input, const Tensor& target, int64_t reduction, double delta, Tensor& output) {
  TORCH_CHECK_NOT_IMPLEMENTED(input.scalar_type() != kLong, "MPS doesn't know how to do square_i64");
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(input.scalar_type()),
                              "huber_loss for complex is not supported for MPS");
  TORCH_CHECK(delta > 0, "huber_loss does not support non-positive values for delta.");
  TORCH_CHECK(target.is_same_size(input), __func__, ": target and input tensors must have identical shapes");
  TORCH_CHECK(output.is_mps());
  if (reduction == Reduction::None)
    output.resize_(target.sizes());
  else
    output.resize_({});
  mps::smooth_huber_fwd_metal(input, target, reduction, static_cast<float>(delta), 1u, output);
  return output;
}

Tensor huber_loss_mps(const Tensor& input, const Tensor& target, int64_t reduction, double delta) {
  TORCH_CHECK(delta > 0, "huber_loss does not support non-positive values for delta.");
  Tensor output = at::empty(input.sizes(), input.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  return huber_loss_out_mps(input, target, reduction, delta, output);
}

Tensor& huber_loss_backward_out_mps(const Tensor& grad_output,
                                    const Tensor& input,
                                    const Tensor& target,
                                    int64_t reduction,
                                    double delta,
                                    Tensor& grad_input) {
  mps::smooth_huber_bwd_metal(grad_output, input, target, reduction, static_cast<float>(delta), 1u, grad_input);
  return grad_input;
}

// MSELoss
TORCH_IMPL_FUNC(mse_loss_out_mps)(const Tensor& input, const Tensor& target, int64_t reduction, const Tensor& output_) {
  using namespace mps;
  if ((input.numel() == 0) || (target.numel() == 0)) {
    reduction == Reduction::Mean ? output_.fill_(std::numeric_limits<float>::quiet_NaN()) : output_.zero_();
    return;
  }
  TORCH_CHECK(target.is_same_size(input), __func__, ": target and input tensors must have identical shapes");
  TORCH_CHECK(c10::isFloatingType(input.scalar_type()) && c10::isFloatingType(target.scalar_type()),
              __func__,
              ": only defined for floating types");
  TORCH_CHECK(output_.is_mps());
  mse_fwd_metal(input.contiguous(), target.contiguous(), reduction, output_);
}

Tensor& mse_loss_backward_out_mps(const Tensor& grad_output,
                                  const Tensor& input,
                                  const Tensor& target,
                                  int64_t reduction,
                                  Tensor& grad_input) {
  return mps::mse_loss_backward_out_impl(grad_output, input, target, reduction, grad_input, __func__);
}

Tensor mse_loss_backward_mps(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction) {
  Tensor grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return mps::mse_loss_backward_out_impl(grad_output, input, target, reduction, grad_input, __func__);
}

// BCELoss
Tensor& binary_cross_entropy_out_mps(const Tensor& input,
                                     const Tensor& target,
                                     const std::optional<Tensor>& weight_opt,
                                     int64_t reduction,
                                     Tensor& loss) {
  return mps::bce_loss_metal(input, target, weight_opt, reduction, loss, std::nullopt, __func__);
}

Tensor binary_cross_entropy_mps(const Tensor& input,
                                const Tensor& target,
                                const std::optional<Tensor>& weight_opt,
                                int64_t reduction) {
  Tensor loss = at::empty_like(input);
  return mps::bce_loss_metal(input, target, weight_opt, reduction, loss, std::nullopt, __func__);
}

Tensor& binary_cross_entropy_backward_out_mps(const Tensor& grad_output,
                                              const Tensor& input,
                                              const Tensor& target,
                                              const std::optional<Tensor>& weight_opt,
                                              int64_t reduction,
                                              Tensor& grad_input) {
  return mps::bce_loss_metal(input, target, weight_opt, reduction, grad_input, grad_output, __func__);
}

Tensor binary_cross_entropy_backward_mps(const Tensor& grad_output,
                                         const Tensor& input,
                                         const Tensor& target,
                                         const std::optional<Tensor>& weight_opt,
                                         int64_t reduction) {
  Tensor grad_input = at::empty_like(input);
  return mps::bce_loss_metal(input, target, weight_opt, reduction, grad_input, grad_output, __func__);
}

// SmoothL1Loss
TORCH_IMPL_FUNC(smooth_l1_loss_out_mps)
(const Tensor& input, const Tensor& target, int64_t reduction, double beta, const Tensor& result) {
  mps::smooth_l1_loss_template(input, target, reduction, beta, result);
}

Tensor& smooth_l1_loss_backward_out_mps(const Tensor& grad_output,
                                        const Tensor& input,
                                        const Tensor& target,
                                        int64_t reduction,
                                        double beta,
                                        Tensor& grad_input) {
  mps::smooth_l1_loss_backward_impl(grad_output, input, target, reduction, beta, grad_input);

  return grad_input;
}

// NLLLoss
TORCH_IMPL_FUNC(nll_loss_backward_out_mps)
(const Tensor& grad_output,
 const Tensor& self,
 const Tensor& target,
 OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& total_weight,
 const Tensor& grad_input) {
  const Tensor& weight = weight_opt.getTensorRef();

  mps::nll_loss_bwd_metal(
      (Tensor&)grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
  return;
}

TORCH_IMPL_FUNC(nll_loss_forward_out_mps)
(const Tensor& self,
 const Tensor& target,
 const OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& output,
 const Tensor& total_weight) {
  const Tensor& weight = weight_opt.getTensorRef();

  mps::nll_loss_fwd_metal((Tensor&)output, (Tensor&)total_weight, self, target, weight, reduction, ignore_index);

  return;
}

inline void check_inputs_nll_loss2d(const Tensor& input, const Tensor& target, const Tensor& weight) {
  TORCH_CHECK(target.dim() == 3,
              "only batches of spatial targets supported (3D tensors)"
              " but got targets of dimension: ",
              target.dim());
  TORCH_CHECK(input.dim() == 4,
              "only batches of spatial inputs supported (4D tensors), "
              "but got input of dimension: ",
              input.dim());
  TORCH_CHECK(!weight.defined() || weight.numel() == input.size(1),
              "weight tensor should be defined either for all or no classes");

  const int64_t input0 = input.size(0);
  const int64_t input2 = input.size(2);
  const int64_t input3 = input.size(3);
  const int64_t target0 = target.size(0);
  const int64_t target1 = target.size(1);
  const int64_t target2 = target.size(2);
  TORCH_CHECK(input0 == target0 && input2 == target1 && input3 == target2,
              "size mismatch (got input: ",
              input.sizes(),
              " , target: ",
              target.sizes());
}

static void nll_loss2d_forward_out_mps_template(Tensor& output,
                                                Tensor& total_weight,
                                                const Tensor& input,
                                                const Tensor& target,
                                                const Tensor& weight,
                                                int64_t reduction,
                                                int64_t ignore_index) {
  check_inputs_nll_loss2d(input, target, weight);
  total_weight.resize_({});

  mps::nllnd_loss_forward_impl(output, total_weight, input, target, weight, reduction, ignore_index, true);

  return;
}

std::tuple<Tensor&, Tensor&> nll_loss2d_forward_out_mps(const Tensor& self,
                                                        const Tensor& target,
                                                        const std::optional<Tensor>& weight_opt,
                                                        int64_t reduction,
                                                        int64_t ignore_index,
                                                        Tensor& output,
                                                        Tensor& total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  nll_loss2d_forward_out_mps_template(output, total_weight, self, target, weight, reduction, ignore_index);
  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

std::tuple<Tensor, Tensor> nll_loss2d_forward_mps(const Tensor& self,
                                                  const Tensor& target,
                                                  const std::optional<Tensor>& weight_opt,
                                                  int64_t reduction,
                                                  int64_t ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  auto output = at::empty({0}, self.options());
  auto total_weight = at::empty({0}, self.options());
  at::native::nll_loss2d_forward_out_mps(self, target, weight, reduction, ignore_index, output, total_weight);
  return std::make_tuple(output, total_weight);
}

static void nll_loss2d_backward_out_mps_template(Tensor& grad_input,
                                                 const Tensor& grad_output,
                                                 const Tensor& input,
                                                 const Tensor& target,
                                                 const Tensor& weight,
                                                 int64_t reduction,
                                                 int64_t ignore_index,
                                                 const Tensor& total_weight) {
  check_inputs_nll_loss2d(input, target, weight);
  grad_input.resize_as_(input);
  grad_input.zero_();
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  TORCH_CHECK(total_weight.numel() == 1,
              "expected total_weight to be a single element tensor, got: ",
              total_weight.sizes(),
              " (",
              total_weight.numel(),
              " elements)");

  mps::nllnd_loss_backward_impl(
      grad_input, grad_output, input, target, weight, reduction, ignore_index, total_weight, true);

  return;
}

Tensor& nll_loss2d_backward_out_mps(const Tensor& grad_output,
                                    const Tensor& self,
                                    const Tensor& target,
                                    const std::optional<Tensor>& weight_opt,
                                    int64_t reduction,
                                    int64_t ignore_index,
                                    const Tensor& total_weight,
                                    Tensor& grad_input) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  nll_loss2d_backward_out_mps_template(
      grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
  return grad_input;
}

Tensor nll_loss2d_backward_mps(const Tensor& grad_output,
                               const Tensor& self,
                               const Tensor& target,
                               const std::optional<Tensor>& weight_opt,
                               int64_t reduction,
                               int64_t ignore_index,
                               const Tensor& total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  auto grad_input = at::zeros_like(self);
  nll_loss2d_backward_out_mps(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
  return grad_input;
}

// =============================================================================
// AutogradMPS overrides — dual-path fusion for huber/smooth_l1/mse losses.
// For reduction=None and N >= kFusionMinNumel, the fused fwd_sg+bwd_sg pair
// halves bwd memory traffic by caching the gradient factor. Otherwise we
// take the standard structured kernel path (already optimized for mean/sum
// via fwd_reduce, which accumulates in fp32 — overflow-safe for fp16/bf16).
// =============================================================================
namespace {

static constexpr int64_t kFusionMinNumel = 1 << 20; // 1,048,576

struct HuberLossMPSAutograd : public torch::autograd::Function<HuberLossMPSAutograd> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            const at::Tensor& input,
                            const at::Tensor& target,
                            int64_t reduction,
                            double delta) {
    at::AutoDispatchBelowAutograd guard;
    if (reduction == at::Reduction::None && input.numel() >= kFusionMinNumel) {
      auto [loss, saved_dg] =
          at::native::mps::huber_or_sl1_fwd_metal_sg(input, target, reduction, static_cast<float>(delta), 1u);
      ctx->save_for_backward({saved_dg});
      ctx->saved_data["reduction"] = reduction;
      ctx->saved_data["delta"] = delta;
      ctx->saved_data["fused"] = true;
      return loss;
    }
    auto loss = at::huber_loss(input, target, reduction, delta);
    ctx->save_for_backward({input, target});
    ctx->saved_data["reduction"] = reduction;
    ctx->saved_data["delta"] = delta;
    ctx->saved_data["fused"] = false;
    return loss;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grads) {
    auto saved = ctx->get_saved_variables();
    auto reduction = ctx->saved_data["reduction"].toInt();
    auto delta = ctx->saved_data["delta"].toDouble();
    at::AutoDispatchBelowAutograd guard;
    if (ctx->saved_data["fused"].toBool()) {
      auto grad_in =
          at::native::mps::huber_or_sl1_bwd_metal_sg(grads[0], saved[0], reduction, static_cast<float>(delta), 1u);
      return {grad_in, ctx->needs_input_grad(1) ? grad_in.neg() : at::Tensor(), at::Tensor(), at::Tensor()};
    }
    auto grad_in = at::huber_loss_backward(grads[0], saved[0], saved[1], reduction, delta);
    at::Tensor grad_target;
    if (ctx->needs_input_grad(1)) {
      grad_target = at::huber_loss_backward(grads[0], saved[1], saved[0], reduction, delta);
    }
    return {grad_in, grad_target, at::Tensor(), at::Tensor()};
  }
};

struct SmoothL1LossMPSAutograd : public torch::autograd::Function<SmoothL1LossMPSAutograd> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            const at::Tensor& input,
                            const at::Tensor& target,
                            int64_t reduction,
                            double beta) {
    at::AutoDispatchBelowAutograd guard;
    if (reduction == at::Reduction::None && input.numel() >= kFusionMinNumel) {
      auto [loss, saved_dg] =
          at::native::mps::huber_or_sl1_fwd_metal_sg(input, target, reduction, static_cast<float>(beta), 0u);
      ctx->save_for_backward({saved_dg});
      ctx->saved_data["reduction"] = reduction;
      ctx->saved_data["beta"] = beta;
      ctx->saved_data["fused"] = true;
      return loss;
    }
    auto loss = at::smooth_l1_loss(input, target, reduction, beta);
    ctx->save_for_backward({input, target});
    ctx->saved_data["reduction"] = reduction;
    ctx->saved_data["beta"] = beta;
    ctx->saved_data["fused"] = false;
    return loss;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grads) {
    auto saved = ctx->get_saved_variables();
    auto reduction = ctx->saved_data["reduction"].toInt();
    auto beta = ctx->saved_data["beta"].toDouble();
    at::AutoDispatchBelowAutograd guard;
    if (ctx->saved_data["fused"].toBool()) {
      auto grad_in =
          at::native::mps::huber_or_sl1_bwd_metal_sg(grads[0], saved[0], reduction, static_cast<float>(beta), 0u);
      return {grad_in, ctx->needs_input_grad(1) ? grad_in.neg() : at::Tensor(), at::Tensor(), at::Tensor()};
    }
    auto grad_in = at::smooth_l1_loss_backward(grads[0], saved[0], saved[1], reduction, beta);
    at::Tensor grad_target;
    if (ctx->needs_input_grad(1)) {
      grad_target = at::smooth_l1_loss_backward(grads[0], saved[1], saved[0], reduction, beta);
    }
    return {grad_in, grad_target, at::Tensor(), at::Tensor()};
  }
};

struct MSELossMPSAutograd : public torch::autograd::Function<MSELossMPSAutograd> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            const at::Tensor& input,
                            const at::Tensor& target,
                            int64_t reduction) {
    at::AutoDispatchBelowAutograd guard;
    if (reduction == at::Reduction::None && input.numel() >= kFusionMinNumel) {
      auto [loss, saved_dg] = at::native::mps::mse_fwd_metal_sg(input, target);
      ctx->save_for_backward({saved_dg});
      ctx->saved_data["reduction"] = reduction;
      ctx->saved_data["fused"] = true;
      return loss;
    }
    auto loss = at::mse_loss(input, target, reduction);
    ctx->save_for_backward({input, target});
    ctx->saved_data["reduction"] = reduction;
    ctx->saved_data["fused"] = false;
    return loss;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grads) {
    auto saved = ctx->get_saved_variables();
    auto reduction = ctx->saved_data["reduction"].toInt();
    at::AutoDispatchBelowAutograd guard;
    if (ctx->saved_data["fused"].toBool()) {
      auto grad_in = at::native::mps::mse_bwd_metal_sg(grads[0], saved[0], reduction);
      return {grad_in, ctx->needs_input_grad(1) ? grad_in.neg() : at::Tensor(), at::Tensor()};
    }
    auto grad_in = at::mse_loss_backward(grads[0], saved[0], saved[1], reduction);
    at::Tensor grad_target;
    if (ctx->needs_input_grad(1)) {
      grad_target = at::mse_loss_backward(grads[0], saved[1], saved[0], reduction);
    }
    return {grad_in, grad_target, at::Tensor()};
  }
};

TORCH_LIBRARY_IMPL(aten, AutogradMPS, m) {
  m.impl("huber_loss",
         [](const at::Tensor& input, const at::Tensor& target, int64_t reduction, double delta) -> at::Tensor {
           TORCH_CHECK(delta > 0, "huber_loss does not support non-positive values for delta.");
           if (input.requires_grad() || target.requires_grad()) {
             return HuberLossMPSAutograd::apply(input, target, reduction, delta);
           }
           at::AutoDispatchBelowAutograd guard;
           return at::huber_loss(input, target, reduction, delta);
         });
  m.impl("smooth_l1_loss",
         [](const at::Tensor& input, const at::Tensor& target, int64_t reduction, double beta) -> at::Tensor {
           TORCH_CHECK(beta >= 0, "smooth_l1_loss does not support negative values for beta.");
           if (input.requires_grad() || target.requires_grad()) {
             return SmoothL1LossMPSAutograd::apply(input, target, reduction, beta);
           }
           at::AutoDispatchBelowAutograd guard;
           return at::smooth_l1_loss(input, target, reduction, beta);
         });
  m.impl("mse_loss", [](const at::Tensor& input, const at::Tensor& target, int64_t reduction) -> at::Tensor {
    if (input.requires_grad() || target.requires_grad()) {
      return MSELossMPSAutograd::apply(input, target, reduction);
    }
    at::AutoDispatchBelowAutograd guard;
    return at::mse_loss(input, target, reduction);
  });
}

} // anonymous namespace

} // namespace at::native
