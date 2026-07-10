//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/LossOps.h>
#include <ATen/native/mps/kernels/ReduceOps.h>
#include <ATen/native/mps/operations/BinaryKernel.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_ctc_loss_backward_native.h>
#include <ATen/ops/_ctc_loss_native.h>
#include <ATen/ops/binary_cross_entropy_backward_native.h>
#include <ATen/ops/binary_cross_entropy_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/huber_loss_backward_native.h>
#include <ATen/ops/huber_loss_native.h>
#include <ATen/ops/mse_loss_backward_native.h>
#include <ATen/ops/mse_loss_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/nll_loss2d_backward_native.h>
#include <ATen/ops/nll_loss2d_forward_native.h>
#include <ATen/ops/nll_loss_backward_native.h>
#include <ATen/ops/nll_loss_forward_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/smooth_l1_loss_backward_native.h>
#include <ATen/ops/smooth_l1_loss_native.h>
#include <ATen/ops/tensor.h>
#endif

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
static auto& reduce_lib = MetalShaderLibrary::getBundledLibrary();
#else
namespace loss_ops_metal {
#include <ATen/native/mps/LossOps_metallib.h>
} // namespace loss_ops_metal
static auto& lib = loss_ops_metal::lib;
namespace reduce_ops_metal {
#include <ATen/native/mps/ReduceOps_metallib.h>
} // namespace reduce_ops_metal
static auto& reduce_lib = reduce_ops_metal::lib;
#endif

// Native Metal MSE loss path replaces the per-shape MPSGraph cache.

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

// Generic two-stage fused pointwise-loss reduction (mean/sum) for mse/bce/
// smooth_l1/huber: pass 1 emits float32 partials, pass 2 reuses sum_reduction.
static constexpr uint32_t kFusedLossThreadsPerTG = 1024;
// Bound below UINT32_MAX by one full grid's vec4 stride
// (MAX_THREADGROUP_SIZE * kFusedLossThreadsPerTG * 4) so the kernels'
// uint32 index math cannot wrap; larger inputs error out or fall back.
static constexpr uint32_t kMaxFusedLossNumel =
    std::numeric_limits<uint32_t>::max() - MAX_THREADGROUP_SIZE * kFusedLossThreadsPerTG * 4;

static void fused_loss_reduce(const std::string& op,
                              const Tensor& input,
                              const Tensor& target,
                              const std::optional<Tensor>& weight,
                              const Tensor& output,
                              FusedLossParams params) {
  // Pass 1 reads input and target as one dtype; promote both (and any weight) to
  // the result dtype so a mixed fp16/fp32 input/target pair is not misread.
  const auto dt = output.scalar_type();
  auto in = input.to(dt);
  auto tgt = target.to(dt);
  std::optional<Tensor> w;
  if (weight.has_value() && weight->defined()) {
    w = weight->expand(input.sizes()).to(dt).contiguous();
  }

  // A reduction is order-free, so when input and target share one dense
  // layout (e.g. both transposed the same way) the kernel reads both buffers
  // in physical order: element pairs still line up and the linear vec4 loads
  // run at contiguous speed with no materialization. Mismatched or gappy
  // layouts fall back to contiguous copies.
  // (Weighted ops stay materialized: a physical-order walk would pair the
  // contiguous weight against permuted input elements.)
  const bool same_dense_layout = !w.has_value() && !in.is_contiguous() && in.strides().equals(tgt.strides()) &&
      in.is_non_overlapping_and_dense() && tgt.is_non_overlapping_and_dense();
  if (!same_dense_layout) {
    in = in.contiguous();
    tgt = tgt.contiguous();
  }

  TORCH_CHECK(in.numel() <= kMaxFusedLossNumel, "fused_loss_reduce: numel exceeds 32-bit kernel indexing");
  const uint32_t numel = static_cast<uint32_t>(in.numel());
  params.numel = numel;
  params.has_weight = w.has_value() ? 1u : 0u;
  // vec4 fast path is only valid when every operand's storage offset keeps the
  // 16-byte alignment the reinterpret_cast<T4*> loads assume.
  params.aligned = (in.storage_offset() % 4 == 0) && (tgt.storage_offset() % 4 == 0) &&
      (!w.has_value() || w->storage_offset() % 4 == 0);

  // Pass 2 reduces the partials in a single threadgroup, so cap the pool at the
  // max threadgroup size.
  uint32_t num_groups = (numel + kFusedLossThreadsPerTG - 1) / kFusedLossThreadsPerTG;
  num_groups = std::clamp<uint32_t>(num_groups, 1u, MAX_THREADGROUP_SIZE);

  Tensor partials = at::empty({static_cast<int64_t>(num_groups)}, in.options().dtype(kFloat));

  // sum_reduction over the 1-D partials -> scalar output; p = numel folds the
  // Mean divide in float32 (p = 0 for Sum leaves the accumulator untouched).
  NormParams<> p2{};
  p2.ndim = 1;
  p2.reduction_size = num_groups;
  p2.input_sizes[0] = num_groups;
  p2.input_strides[0] = 1;
  p2.output_sizes[0] = 1;
  p2.output_strides[0] = 0;
  p2.p = (params.reduction == Reduction::Mean) ? static_cast<float>(numel) : 0.0f;

  const std::string p1name = fmt::format("fused_loss_pass1_{}_{}", op, scalarToMetalTypeString(in));
  const std::string p2name = fmt::format("sum_reduction_float_{}", scalarToMetalTypeString(output));

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> ce = stream->commandEncoder();

      auto ps1 = lib.getPipelineStateForFunc(p1name);
      getMPSProfiler().beginProfileKernel(ps1, op + "_fused_pass1", {in, tgt});
      [ce setComputePipelineState:ps1];
      mtl_setArgs(ce, in, tgt, w, partials, params); // w nulls buffer(2) when absent
      [ce dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kFusedLossThreadsPerTG, 1, 1)];
      getMPSProfiler().endProfileKernel(ps1);

      auto ps2 = reduce_lib.getPipelineStateForFunc(p2name);
      getMPSProfiler().beginProfileKernel(ps2, "sum_reduction_pass2", {partials});
      [ce setComputePipelineState:ps2];
      mtl_setArgs(ce, partials, output, p2);
      uint32_t tpg2 = std::min<uint32_t>(MAX_THREADGROUP_SIZE, ((num_groups + 31u) / 32u) * 32u);
      tpg2 = std::max<uint32_t>(tpg2, 32u);
      [ce dispatchThreads:MTLSizeMake(tpg2, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg2, 1, 1)];
      getMPSProfiler().endProfileKernel(ps2);
    }
  });
}

// Dense fast path for the fused elementwise loss backward: one vec4 kernel
// computing Op::bwd over input/target/grad_output when everything shares one
// dtype and one dense layout (a reduction-free elementwise map is layout-
// agnostic when all operands, including the output, agree on strides).
// Returns false when the combination needs the generic TensorIterator
// ternary fallback.
static bool fused_loss_bwd_fast_path(const std::string& op,
                                     const Tensor& input,
                                     const Tensor& target,
                                     const Tensor& grad_output,
                                     const Tensor& grad_input,
                                     double norm,
                                     double p1 = 0.0) {
  const auto dt = input.scalar_type();
  if (!c10::isFloatingType(dt) || dt == kDouble) {
    return false;
  }
  if (target.scalar_type() != dt || grad_input.scalar_type() != dt || grad_output.scalar_type() != dt) {
    return false;
  }
  const bool scalar_grad = grad_output.dim() == 0;
  // Sizes must match too: a same-strides but smaller grad_input (possible
  // when the op is called directly rather than via autograd) would otherwise
  // pass the stride check and the kernel would write numel elements out of
  // bounds.
  auto dense_like_input = [&](const Tensor& t) {
    return t.is_non_overlapping_and_dense() && t.sizes().equals(input.sizes()) && t.strides().equals(input.strides());
  };
  if (!input.is_non_overlapping_and_dense() || !dense_like_input(target) || !dense_like_input(grad_input)) {
    return false;
  }
  if (!scalar_grad && !dense_like_input(grad_output)) {
    return false;
  }
  if (input.numel() > kMaxFusedLossNumel) {
    return false;
  }

  FusedLossParams params{};
  params.numel = static_cast<uint32_t>(input.numel());
  params.flag = scalar_grad ? 1u : 0u;
  params.p0 = static_cast<float>(norm);
  params.p1 = static_cast<float>(p1);
  params.aligned = (input.storage_offset() % 4 == 0) && (target.storage_offset() % 4 == 0) &&
      (grad_input.storage_offset() % 4 == 0) && (scalar_grad || grad_output.storage_offset() % 4 == 0);

  const std::string name = fmt::format("fused_loss_bwd_{}_{}", op, scalarToMetalTypeString(input));
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> ce = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(name);
      getMPSProfiler().beginProfileKernel(pso, op + "_fused_bwd", {input, target, grad_output});
      [ce setComputePipelineState:pso];
      mtl_setArgs(ce, input, target, grad_output, grad_input, params);
      const uint32_t jobs = params.aligned ? (params.numel + 3u) / 4u : params.numel;
      mtl_dispatch1DJob(ce, pso, jobs);
      getMPSProfiler().endProfileKernel(pso);
    }
  });
  return true;
}

static Tensor& mse_loss_backward_out_impl(const Tensor& grad_output,
                                          const Tensor& input,
                                          const Tensor& target,
                                          int64_t reduction,
                                          Tensor& grad_input,
                                          const std::string& op_name) {
  // CPU broadcasts input/target/grad_output through a TensorIterator; match
  // that instead of requiring identical shapes. Only broadcastability is
  // validated here (infer_size raises the standard size-mismatch error):
  // the dense fast path below self-rejects mismatched sizes and the ternary
  // iterator fallback broadcasts natively.
  if (!target.is_same_size(input)) {
    (void)at::infer_size_dimvector(input.sizes(), target.sizes());
  }
  auto norm = reduction == Reduction::Mean ? 2. / static_cast<double>(input.numel()) : 2.;

  // Empty input: gradient norm is 2/numel for mean (-> NaN as 2/0), else 0.
  // Same-size only: an empty-via-broadcast pair falls through to the
  // ternary iterator, which resizes grad_input to the (empty) common shape
  // exactly like the CPU TensorIterator.
  if (target.is_same_size(input) && ((input.numel() == 0) || (grad_output.numel() == 0))) {
    reduction == Reduction::Mean ? grad_input.fill_(std::numeric_limits<float>::quiet_NaN()) : grad_input.zero_();
    return grad_input;
  }

  // grad_input = norm * (input - target) * grad_output in ONE fused pass (no
  // MPSGraph, no materialized intermediate). For mean, match the CPU
  // kernel's rounding: norm (2/N) is narrowed to the compute dtype
  // (value.to<scalar_t>()) before it is applied -- for fp16 that rounding is
  // visible in the gradients, and 2/N can flush to zero for huge N exactly
  // as it does on CPU.
  if (reduction == Reduction::Mean) {
    const auto common = at::promoteTypes(at::result_type(input, target), grad_output.scalar_type());
    if (common == kHalf) {
      norm = static_cast<double>(static_cast<c10::Half>(norm));
    } else if (common == kBFloat16) {
      norm = static_cast<double>(static_cast<c10::BFloat16>(norm));
    } else if (common == kFloat) {
      norm = static_cast<double>(static_cast<float>(norm));
    }
  }

  if (fused_loss_bwd_fast_path("mse", input, target, grad_output, grad_input, norm)) {
    return grad_input;
  }

  // Generic fallback: the TensorIterator ternary handles mixed dtypes (with
  // a half input and a float target the subtract runs in the promoted
  // float, fixing the mixed-dtype case the parent MPSGraph path hard-crashed
  // on), broadcast scalar grads, and mismatched layouts. norm is exactly 2
  // for reduction none/sum, baked into mse_backward2; for mean the 0-dim
  // grad_output is pre-scaled by the narrowed norm in float (in the
  // gradient's own dtype 2/N can underflow fp16/bf16; non-inplace because
  // .to(kFloat) is a no-op for a float grad_output and mul_ would mutate the
  // caller's tensor).
  Tensor grad = grad_output;
  std::string kernel = "mse_backward2";
  if (reduction == Reduction::Mean) {
    grad = grad_output.to(kFloat).mul(norm);
    kernel = "mse_backward_scaled";
  }
  ternary_op_kernel(kernel, input, target, grad, grad_input);
  return grad_input;
}

// namespace to localize the CachedGraph struct for Binary Cross Entropy
namespace BCELoss {

struct CachedGraph : public MPSCachedGraph {
  CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor *inputTensor = nil, *targetTensor = nil;
  // gradOutput only used on backward pass
  MPSGraphTensor *weightTensor = nil, *gradOutputTensor = nil;
  // lossTensor used for forward, and gradInputTensor for backward pass
  union {
    MPSGraphTensor* lossTensor = nil;
    MPSGraphTensor* gradInputTensor;
  };
};

static MPSGraphTensor* bce_forward_mps(CachedGraph* bceGraph) {
  MPSGraph* mpsGraph = bceGraph->graph();
  const auto inputType = [bceGraph->inputTensor dataType];

  // Forward BCE: L = -w (y ln(x) + (1-y) ln(1-x))
  MPSGraphTensor* one = [mpsGraph constantWithScalar:1.0 dataType:inputType];
  // -100 is the hard limit value defined in BCELoss Spec. to clamp the log
  MPSGraphTensor* neg100 = [mpsGraph constantWithScalar:-100.0 dataType:inputType];
  // 1 - x
  MPSGraphTensor* one_Input = [mpsGraph subtractionWithPrimaryTensor:one
                                                     secondaryTensor:bceGraph->inputTensor
                                                                name:nil];
  // log(x)
  MPSGraphTensor* logInput = [mpsGraph logarithmWithTensor:bceGraph->inputTensor name:nil];
  // max(log(x), -100)
  MPSGraphTensor* clampedLogInput = [mpsGraph maximumWithPrimaryTensor:logInput secondaryTensor:neg100 name:nil];
  // log(1 - x)
  MPSGraphTensor* log1_Input = [mpsGraph logarithmWithTensor:one_Input name:nil];
  // max(log(1 - x), -100)
  MPSGraphTensor* clampedLog1_Input = [mpsGraph maximumWithPrimaryTensor:log1_Input secondaryTensor:neg100 name:nil];
  // (y - 1) resulted from -(1 - y)
  MPSGraphTensor* target_1 = [mpsGraph subtractionWithPrimaryTensor:bceGraph->targetTensor
                                                    secondaryTensor:one
                                                               name:nil];
  // (y - 1) * max(log(1 - x), -100)
  MPSGraphTensor* target_1TimesLog1_Input = [mpsGraph multiplicationWithPrimaryTensor:target_1
                                                                      secondaryTensor:clampedLog1_Input
                                                                                 name:nil];
  // y * max(log(x), -100)
  MPSGraphTensor* targetTimesLogInput = [mpsGraph multiplicationWithPrimaryTensor:bceGraph->targetTensor
                                                                  secondaryTensor:clampedLogInput
                                                                             name:nil];
  // ((y - 1) * max(log(1 - x), -100)) - (y * max(log(x), -100))
  MPSGraphTensor* bceLoss = [mpsGraph subtractionWithPrimaryTensor:target_1TimesLog1_Input
                                                   secondaryTensor:targetTimesLogInput
                                                              name:nil];
  return bceLoss;
}

static MPSGraphTensor* bce_backward_mps(CachedGraph* bceGraph) {
  MPSGraph* mpsGraph = bceGraph->graph();
  const auto inputType = [bceGraph->inputTensor dataType];

  // Backward BCE: d(L)/d(x) = -w (y - x) / (x - x^2)
  MPSGraphTensor* one = [mpsGraph constantWithScalar:1.0 dataType:inputType];
  // epsilon used to clamp the grad input denominator
  MPSGraphTensor* epsilon = [mpsGraph constantWithScalar:1e-12 dataType:inputType];
  // 1 - x
  MPSGraphTensor* one_Input = [mpsGraph subtractionWithPrimaryTensor:one
                                                     secondaryTensor:bceGraph->inputTensor
                                                                name:nil];
  // x * (1 - x)
  MPSGraphTensor* inputTimes1_Input = [mpsGraph multiplicationWithPrimaryTensor:bceGraph->inputTensor
                                                                secondaryTensor:one_Input
                                                                           name:nil];
  // max(x * (1 - x), epsilon)
  MPSGraphTensor* gradInputDenominator = [mpsGraph maximumWithPrimaryTensor:inputTimes1_Input
                                                            secondaryTensor:epsilon
                                                                       name:nil];
  // (x - y)
  MPSGraphTensor* input_target = [mpsGraph subtractionWithPrimaryTensor:bceGraph->inputTensor
                                                        secondaryTensor:bceGraph->targetTensor
                                                                   name:nil];
  // (x - y) / max(x * (1 - x), epsilon)
  MPSGraphTensor* inputDivGradInputDenom = [mpsGraph divisionWithPrimaryTensor:input_target
                                                               secondaryTensor:gradInputDenominator
                                                                          name:nil];
  // gradOutput * (((x - y) / max(x * (1 - x), epsilon)))
  MPSGraphTensor* gradInput = [mpsGraph multiplicationWithPrimaryTensor:bceGraph->gradOutputTensor
                                                        secondaryTensor:inputDivGradInputDenom
                                                                   name:nil];
  return gradInput;
}

// Binary Cross Enropy (Forward/Backward BCELoss)
// NOTE: "loss" tensor would be "grad_input" if it's a backward pass
static Tensor& bce_loss_out_impl(const Tensor& input,
                                 const Tensor& target,
                                 const std::optional<Tensor>& weight_opt,
                                 int64_t reduction,
                                 Tensor& loss,
                                 const std::optional<Tensor>& grad_output_opt,
                                 const std::string& op_name) {
  // TODO: add sanity check for the elements of input tensor to be within [0..1]
  TORCH_CHECK(target.is_same_size(input), op_name + ": target and input tensors must have identical shapes")

  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  c10::MaybeOwned<Tensor> grad_output_maybe_owned = at::borrow_from_optional_tensor(grad_output_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& grad_output = *grad_output_maybe_owned;

  loss.resize_((reduction == Reduction::None || grad_output.defined()) ? target.sizes() : IntArrayRef({}));
  TORCH_CHECK(loss.is_mps());

  @autoreleasepool {
    std::string key = op_name + reductionToString(reduction) + getTensorsStringKey({input, target, weight});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
      newCachedGraph->targetTensor = mpsGraphRankedPlaceHolder(mpsGraph, target);

      MPSGraphTensor* bceLossUnweighted = nil;
      // if grad_output is defined, then it's a backward pass
      if (grad_output.defined()) {
        newCachedGraph->gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
        bceLossUnweighted = bce_backward_mps(newCachedGraph);
      } else {
        bceLossUnweighted = bce_forward_mps(newCachedGraph);
      }

      MPSGraphTensor* bceLoss = bceLossUnweighted;
      if (weight.defined()) {
        newCachedGraph->weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight);
        bceLoss = [mpsGraph multiplicationWithPrimaryTensor:bceLossUnweighted
                                            secondaryTensor:newCachedGraph->weightTensor
                                                       name:nil];
      }

      if (grad_output.defined()) {
        if (reduction == at::Reduction::Mean) {
          MPSGraphTensor* inputNumel = [mpsGraph constantWithScalar:static_cast<double>(input.numel())
                                                           dataType:[bceLoss dataType]];
          newCachedGraph->gradInputTensor = [mpsGraph divisionWithPrimaryTensor:bceLoss
                                                                secondaryTensor:inputNumel
                                                                           name:nil];
        } else {
          newCachedGraph->gradInputTensor = bceLoss;
        }
      } else {
        newCachedGraph->lossTensor = reduceTensor(bceLoss, reduction, mpsGraph, input.sizes().size());
      }
    });
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor, input);
    Placeholder targetPlaceholder = Placeholder(cachedGraph->targetTensor, target);
    Placeholder lossPlaceholder = Placeholder(cachedGraph->lossTensor, loss);

    NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];

    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[targetPlaceholder.getMPSGraphTensor()] = targetPlaceholder.getMPSGraphTensorData();
    if (weight.defined()) {
      Placeholder weightPlaceholder = Placeholder(cachedGraph->weightTensor, weight);
      feeds[weightPlaceholder.getMPSGraphTensor()] = weightPlaceholder.getMPSGraphTensorData();
    }
    if (grad_output.defined()) {
      Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor, grad_output);
      feeds[gradOutputPlaceholder.getMPSGraphTensor()] = gradOutputPlaceholder.getMPSGraphTensorData();
    }

    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, lossPlaceholder);
  }

  return loss;
}

} // namespace BCELoss

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

// Unified native smooth_l1 / huber path (no weight). Forward mean/sum uses
// the shared fused_loss_reduce (SmoothL1Op/HuberOp, beta/delta in p0);
// reduction=None uses the smooth_l1/huber binary-alpha iterator kernels
// (alpha carries beta/delta). Backward rides fused_loss_bwd's dense fast
// path (norm in p0, beta/delta in p1) with a two-pass TensorIterator
// fallback (*_backward_clip kernel, then a multiply by the pre-scaled grad)
// for mixed dtypes or mismatched layouts. "out" is grad_input on the
// backward pass.
static Tensor& smooth_huber_native(const Tensor& input,
                                   const Tensor& target,
                                   int64_t reduction,
                                   double beta,
                                   bool is_huber,
                                   Tensor& out,
                                   const std::optional<Tensor>& grad_output_opt,
                                   const std::string& op_name) {
  TORCH_CHECK(c10::isFloatingType(input.scalar_type()) && c10::isFloatingType(target.scalar_type()),
              op_name,
              ": only defined for floating types");
  const bool is_bwd = grad_output_opt.has_value();
  // CPU broadcasts input against target (and grad_output on the backward
  // pass) through a TensorIterator, and the old MPSGraph backward inherited
  // the same semantics from the graph's implicit numpy-style broadcast.
  // Match that instead of requiring identical shapes: infer_size raises the
  // standard size-mismatch error for non-broadcastable shapes, and the
  // result/grad_input is sized from the broadcast (common) shape, never
  // from target alone.
  const auto pair_shape = at::infer_size_dimvector(input.sizes(), target.sizes());
  const auto common_shape = is_bwd ? at::infer_size_dimvector(pair_shape, grad_output_opt->sizes()) : pair_shape;
  const auto out_shape = (reduction == Reduction::None || is_bwd) ? IntArrayRef(common_shape) : IntArrayRef({});
  if (!out.sizes().equals(out_shape)) {
    out.resize_(out_shape);
  }
  TORCH_CHECK(out.is_mps());
  if (input.numel() == 0 || target.numel() == 0) {
    if (!is_bwd) {
      reduction == Reduction::Mean ? out.fill_(std::numeric_limits<float>::quiet_NaN()) : out.zero_();
    }
    return out;
  }

  const char* op = is_huber ? "huber" : "smooth_l1";
  if (!is_bwd && reduction != Reduction::None) {
    FusedLossParams params{};
    params.reduction = static_cast<uint32_t>(reduction);
    params.p0 = static_cast<float>(beta);
    // The fused mean/sum kernel walks one flat dense pair, so materialize
    // numpy-style broadcast views first: expand_outplace borrows when the
    // shapes already match (no copy on the equal-shape hot path), and
    // fused_loss_reduce's contiguous() copies any stride-0 expanded view.
    auto [b_input, b_target] = at::expand_outplace(input, target);
    fused_loss_reduce(op, *b_input, *b_target, std::nullopt, out, params);
    return out;
  }
  if (!is_bwd) {
    binary_op_kernel(op, input, target, out, c10::Scalar(beta));
    return out;
  }

  const double scale = (reduction == Reduction::Mean) ? 1.0 / static_cast<double>(input.numel()) : 1.0;
  if (fused_loss_bwd_fast_path(op, input, target, *grad_output_opt, out, /*norm=*/scale, /*p1=*/beta)) {
    return out;
  }
  // Fallback: clipped-difference term via the binary-alpha iterator, then a
  // multiply by the grad with the mean scale folded in.
  const auto compute_dtype = at::promoteTypes(at::result_type(input, target), grad_output_opt->scalar_type());
  // The clipped-difference term has the input/target broadcast shape; the
  // final mul_out then broadcasts it against the grad.
  Tensor clip = at::empty(pair_shape, input.options().dtype(compute_dtype));
  binary_op_kernel(std::string(op) + "_backward_clip", input, target, clip, c10::Scalar(beta));
  Tensor g_eff = *grad_output_opt;
  if (scale != 1.0) {
    g_eff = g_eff.to(kFloat).mul(scale);
  }
  at::mul_out(out, clip, g_eff);
  return out;
}

} // namespace mps

Tensor& huber_loss_out_mps(const Tensor& input, const Tensor& target, int64_t reduction, double delta, Tensor& output) {
  TORCH_CHECK(delta > 0, "huber_loss does not support non-positive values for delta.")
  return mps::smooth_huber_native(input, target, reduction, delta, /*is_huber=*/true, output, std::nullopt, __func__);
}

Tensor huber_loss_mps(const Tensor& input, const Tensor& target, int64_t reduction, double delta) {
  // For reduction=None, inherit the input layout (like the structured TI
  // ops): a contiguous out against non-contiguous inputs would force the
  // slow strided iterator kernel for no reason.
  Tensor output = (reduction == Reduction::None) ? at::empty_like(input) : at::empty({0}, input.options());
  return huber_loss_out_mps(input, target, reduction, delta, output);
}

Tensor& huber_loss_backward_out_mps(const Tensor& grad_output,
                                    const Tensor& input,
                                    const Tensor& target,
                                    int64_t reduction,
                                    double delta,
                                    Tensor& grad_input) {
  TORCH_CHECK(delta > 0, "huber_loss_backward does not support non-positive values for delta.")
  return mps::smooth_huber_native(
      input, target, reduction, delta, /*is_huber=*/true, grad_input, grad_output, __func__);
}

// MSELoss: reduction=None uses the `mse` binary kernel; mean/sum use the fused
// float32 GPU reduction (no MPSGraph -> no per-shape graph cache).
TORCH_IMPL_FUNC(mse_loss_out_mps)(const Tensor& input, const Tensor& target, int64_t reduction, const Tensor& output_) {
  std::string op_name = "mse_loss_out_mps";
  using namespace mps;
  // Empty input: mean is NaN (0/0), sum and none are 0. Matches CPU/CUDA.
  if ((input.numel() == 0) || (target.numel() == 0)) {
    reduction == Reduction::Mean ? output_.fill_(std::numeric_limits<float>::quiet_NaN()) : output_.zero_();
    return;
  }

  TORCH_CHECK(c10::isFloatingType(input.scalar_type()) && c10::isFloatingType(target.scalar_type()),
              op_name + ": only defined for floating types");
  TORCH_CHECK(output_.is_mps());

  if (reduction == Reduction::None) {
    // (input - target)^2 written straight to the (possibly non-contiguous)
    // structured output. exec_binary_kernel handles strided/broadcast inputs.
    binary_op_kernel("mse", input, target, output_);
    return;
  }

  // Mean/Sum: fused square-and-reduce in a single GPU pass (no materialized
  // squared-difference temp). fused_loss_reduce promotes both operands to the
  // output dtype, the kernel computes (input - target)^2 in float32,
  // threadgroup-reduces, and (for mean) divides by numel in float32 before
  // casting to the output dtype.
  // This matches the fused MPSGraph square+reduce that the materialize-then-
  // at::sum path regressed against, and keeps the float32 accumulation so
  // large-N fp16/bf16 does not overflow. output_ is a scalar tensor.
  // The structured meta has already validated and broadcast the shapes (CPU
  // parity). The fused kernel walks one flat dense pair, so materialize
  // numpy-style broadcast views first: expand_outplace borrows when the
  // shapes already match (no copy on the equal-shape hot path), and
  // fused_loss_reduce's contiguous() copies any stride-0 expanded view.
  auto [b_input, b_target] = at::expand_outplace(input, target);
  FusedLossParams params{};
  params.reduction = static_cast<uint32_t>(reduction);
  fused_loss_reduce("mse", *b_input, *b_target, std::nullopt, output_, params);
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
  return mps::BCELoss::bce_loss_out_impl(input, target, weight_opt, reduction, loss, std::nullopt, __func__);
}

Tensor binary_cross_entropy_mps(const Tensor& input,
                                const Tensor& target,
                                const std::optional<Tensor>& weight_opt,
                                int64_t reduction) {
  Tensor loss = at::empty_like(input);
  return mps::BCELoss::bce_loss_out_impl(input, target, weight_opt, reduction, loss, std::nullopt, __func__);
}

Tensor& binary_cross_entropy_backward_out_mps(const Tensor& grad_output,
                                              const Tensor& input,
                                              const Tensor& target,
                                              const std::optional<Tensor>& weight_opt,
                                              int64_t reduction,
                                              Tensor& grad_input) {
  return mps::BCELoss::bce_loss_out_impl(input, target, weight_opt, reduction, grad_input, grad_output, __func__);
}

Tensor binary_cross_entropy_backward_mps(const Tensor& grad_output,
                                         const Tensor& input,
                                         const Tensor& target,
                                         const std::optional<Tensor>& weight_opt,
                                         int64_t reduction) {
  Tensor grad_input = at::empty_like(input);
  return mps::BCELoss::bce_loss_out_impl(input, target, weight_opt, reduction, grad_input, grad_output, __func__);
}

// SmoothL1Loss
TORCH_IMPL_FUNC(smooth_l1_loss_out_mps)
(const Tensor& input, const Tensor& target, int64_t reduction, double beta, const Tensor& result) {
  TORCH_CHECK(beta >= 0, "smooth_l1_loss does not support negative values for beta.");
  Tensor result_ = const_cast<Tensor&>(result);
  mps::smooth_huber_native(
      input, target, reduction, beta, /*is_huber=*/false, result_, std::nullopt, "smooth_l1_loss_out_mps");
}

Tensor& smooth_l1_loss_backward_out_mps(const Tensor& grad_output,
                                        const Tensor& input,
                                        const Tensor& target,
                                        int64_t reduction,
                                        double beta,
                                        Tensor& grad_input) {
  TORCH_CHECK(beta >= 0, "smooth_l1_loss_backward does not support negative values for beta.");
  return mps::smooth_huber_native(
      input, target, reduction, beta, /*is_huber=*/false, grad_input, grad_output, __func__);
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

  mps::nllnd_loss_backward_impl(
      (Tensor&)grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight, false);
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

  mps::nllnd_loss_forward_impl(
      (Tensor&)output, (Tensor&)total_weight, self, target, weight, reduction, ignore_index, false);

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

template <typename index_t>
std::string_view get_index_type_str() {
  if constexpr (std::is_same_v<index_t, int32_t>) {
    return "int32_t";
  } else if constexpr (std::is_same_v<index_t, int64_t>) {
    return "int64_t";
  } else {
    static_assert(false);
  }
}

template <typename index_t, bool beta = false>
static void ctc_loss_mps_kernel(const std::optional<Tensor>& loss,
                                Tensor& log_alpha,
                                const Tensor& log_probs,
                                const Tensor& targets,
                                const Tensor& input_lengths_t,
                                const Tensor& target_lengths_t,
                                const Tensor& target_batch_offsets_t,
                                int64_t BLANK,
                                int64_t max_input_length,
                                int64_t max_target_length,
                                int64_t batch_size,
                                int64_t tg_target_stride) {
  using namespace mps;
  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> pso = mps::lib.getPipelineStateForFunc(fmt::format("ctc_loss{}_{}_{}_{}",
                                                                                     beta ? "_backward_log_beta" : "",
                                                                                     scalarToMetalTypeString(log_probs),
                                                                                     scalarToMetalTypeString(targets),
                                                                                     get_index_type_str<index_t>()));
      const uint32_t TG_SIZE = std::min<int64_t>([pso maxTotalThreadsPerThreadgroup], 2 * max_target_length + 1);
      [computeEncoder setComputePipelineState:pso];

      CTCLossParams<index_t> params;

      params.max_input_length = max_input_length;
      params.max_target_length = max_target_length;
      params.batch_size = batch_size;
      params.BLANK = BLANK;
      params.tg_target_stride = tg_target_stride;
      params.log_probs_time_stride = log_probs.stride(0);
      params.log_probs_batch_stride = log_probs.stride(1);
      params.log_probs_token_stride = log_probs.stride(2);
      params.log_alpha_batch_stride = log_alpha.stride(0);
      params.log_alpha_time_stride = log_alpha.stride(1);
      params.log_alpha_target_stride = log_alpha.stride(2);

      if constexpr (beta) {
        mtl_setArgs(computeEncoder,
                    log_alpha,
                    log_probs,
                    targets,
                    input_lengths_t,
                    target_lengths_t,
                    target_batch_offsets_t,
                    params);

      } else {
        TORCH_INTERNAL_ASSERT(loss.has_value(), "loss tensor must have a value when beta=false");
        mtl_setArgs(computeEncoder,
                    *loss,
                    log_alpha,
                    log_probs,
                    targets,
                    input_lengths_t,
                    target_lengths_t,
                    target_batch_offsets_t,
                    params);
      }
      [computeEncoder dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
    }
  });
}

static void ctc_loss_mps_check(const Tensor& log_probs,
                               const Tensor& targets,
                               IntArrayRef input_lengths,
                               IntArrayRef target_lengths,
                               int64_t BLANK) {
  TORCH_CHECK(log_probs.dim() == 3, "log_probs must be 3-D (T, N, C)");
  TORCH_CHECK(targets.dim() >= 1 && targets.dim() <= 2, "targets must have 1 or 2 dims");
  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  TORCH_CHECK((0 <= BLANK) && (BLANK < num_labels), "blank must be in label range");
  TORCH_CHECK((int64_t)input_lengths.size() == batch_size, "input_lengths must be of size batch_size");
  TORCH_CHECK((int64_t)target_lengths.size() == batch_size, "target_lengths must be of size batch_size");
  TORCH_CHECK(log_probs.numel() > 0, "log_probs tensor must not be empty");
}

std::tuple<Tensor, Tensor> ctc_loss_mps(const Tensor& log_probs,
                                        const Tensor& targets,
                                        IntArrayRef input_lengths,
                                        IntArrayRef target_lengths,
                                        int64_t BLANK,
                                        bool zero_infinity) {
  using namespace mps;
  ctc_loss_mps_check(log_probs, targets, input_lengths, target_lengths, BLANK);

  int64_t batch_size = log_probs.size(1);

  // Compute per-batch target offsets and max target length
  int64_t tg_target_stride = (targets.dim() == 1) ? targets.stride(0) : targets.stride(1);
  int64_t max_target_length = 0;
  int64_t max_input_length = log_probs.size(0);
  std::vector<int64_t> target_batch_offsets(batch_size);

  int64_t pos = 0;
  int64_t tg_batch_stride = targets.stride(0);
  for (int64_t i = 0; i < batch_size; i++) {
    TORCH_CHECK(target_lengths[i] >= 0,
                "Expected target_lengths to have value at least 0, but got value ",
                target_lengths[i],
                " (while checking arguments for ctc_loss_mps)");
    TORCH_CHECK(input_lengths[i] >= 0 && input_lengths[i] <= max_input_length,
                "Expected input_lengths to be in [0, ",
                max_input_length,
                "], but got value ",
                input_lengths[i],
                " (while checking arguments for ctc_loss_mps)");
    if (targets.dim() == 1) {
      target_batch_offsets[i] = pos;
      pos += target_lengths[i];
    } else {
      target_batch_offsets[i] = i * tg_batch_stride;
    }
    max_target_length = std::max(max_target_length, target_lengths[i]);
  }
  if (targets.dim() == 2) {
    TORCH_CHECK(targets.size(1) >= max_target_length,
                "Expected targets to have size at least ",
                max_target_length,
                " at dimension 1, but got size ",
                targets.size(1),
                " (while checking arguments for ctc_loss_mps)");
  }

  Tensor loss = at::empty({batch_size}, log_probs.options());
  Tensor log_alpha = at::empty({batch_size, log_probs.size(0), 2 * max_target_length + 1}, log_probs.options());

  if (batch_size == 0) {
    return {loss, log_alpha};
  }

  bool can_use_32bit_index_math = at::native::canUse32BitIndexMath(log_probs) &&
      at::native::canUse32BitIndexMath(targets) && at::native::canUse32BitIndexMath(log_alpha);
  // NOTE: Used signed types because if we attempt to use unsigned, the
  // `at::tensor` calls below raise the error: "Exception: "tensor_cpu" not
  // implemented for 'UInt32'"
  auto metadata_dtype = can_use_32bit_index_math ? kInt : kLong;

  // Move metadata to MPS device
  Tensor input_lengths_t = at::tensor(input_lengths, log_probs.options().dtype(metadata_dtype));
  Tensor target_lengths_t = at::tensor(target_lengths, log_probs.options().dtype(metadata_dtype));
  Tensor target_batch_offsets_t = at::tensor(target_batch_offsets, log_probs.options().dtype(metadata_dtype));

  if (can_use_32bit_index_math) {
    ctc_loss_mps_kernel<int32_t>(loss,
                                 log_alpha,
                                 log_probs,
                                 targets,
                                 input_lengths_t,
                                 target_lengths_t,
                                 target_batch_offsets_t,
                                 BLANK,
                                 max_input_length,
                                 max_target_length,
                                 batch_size,
                                 tg_target_stride);
  } else {
    ctc_loss_mps_kernel<int64_t>(loss,
                                 log_alpha,
                                 log_probs,
                                 targets,
                                 input_lengths_t,
                                 target_lengths_t,
                                 target_batch_offsets_t,
                                 BLANK,
                                 max_input_length,
                                 max_target_length,
                                 batch_size,
                                 tg_target_stride);
  }

  return {std::move(loss), std::move(log_alpha)};
}

template <typename index_t>
static void ctc_loss_backward_mps_kernel(Tensor& grad,
                                         const Tensor& grad_out,
                                         const Tensor& log_alpha,
                                         const Tensor& log_probs,
                                         const Tensor& targets,
                                         const Tensor& input_lengths_t,
                                         const Tensor& target_lengths_t,
                                         const Tensor& loss,
                                         const Tensor& target_batch_offsets_t,
                                         int64_t BLANK,
                                         int64_t max_input_length,
                                         int64_t batch_size,
                                         bool zero_infinity) {
  using namespace mps;
  // Derive max_target_length from log_alpha shape (same as CUDA backward).
  int64_t max_target_length = log_alpha.size(2) / 2;
  int64_t tg_target_stride = (targets.dim() == 1) ? targets.stride(0) : targets.stride(1);

  Tensor log_beta = at::empty_like(log_alpha, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  ctc_loss_mps_kernel<index_t, /*beta=*/true>(/*loss=*/std::nullopt,
                                              log_beta,
                                              log_probs,
                                              targets,
                                              input_lengths_t,
                                              target_lengths_t,
                                              target_batch_offsets_t,
                                              BLANK,
                                              max_input_length,
                                              max_target_length,
                                              batch_size,
                                              tg_target_stride);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> pso =
          mps::lib.getPipelineStateForFunc(fmt::format("ctc_loss_backward_collect_{}_{}_{}",
                                                       scalarToMetalTypeString(log_probs),
                                                       scalarToMetalTypeString(targets),
                                                       get_index_type_str<index_t>()));
      [computeEncoder setComputePipelineState:pso];

      CTCLossBackwardCollectParams<index_t> params;
      params.BLANK = BLANK;
      params.max_input_length = max_input_length;
      params.max_target_length = max_target_length;
      params.num_labels = log_probs.size(2);
      params.tg_target_stride = tg_target_stride;
      params.log_probs_time_stride = log_probs.stride(0);
      params.log_probs_batch_stride = log_probs.stride(1);
      params.log_probs_token_stride = log_probs.stride(2);
      params.log_alpha_beta_batch_stride = log_alpha.stride(0);
      params.log_alpha_beta_time_stride = log_alpha.stride(1);
      params.log_alpha_beta_target_stride = log_alpha.stride(2);
      params.grad_time_stride = grad.stride(0);
      params.grad_batch_stride = grad.stride(1);
      params.grad_token_stride = grad.stride(2);
      params.grad_out_batch_stride = grad_out.stride(0);
      params.zero_infinity = zero_infinity;

      mtl_setArgs(computeEncoder,
                  grad,
                  grad_out,
                  log_alpha,
                  log_beta,
                  log_probs,
                  targets,
                  input_lengths_t,
                  target_lengths_t,
                  loss,
                  target_batch_offsets_t,
                  params);
      [computeEncoder
                dispatchThreads:MTLSizeMake(max_input_length, batch_size, 1)
          threadsPerThreadgroup:MTLSizeMake(
                                    std::min<int64_t>([pso maxTotalThreadsPerThreadgroup], max_input_length), 1, 1)];
    }
  });
}

Tensor ctc_loss_backward_mps(const Tensor& grad_out,
                             const Tensor& log_probs,
                             const Tensor& targets,
                             IntArrayRef input_lengths,
                             IntArrayRef target_lengths,
                             const Tensor& loss,
                             const Tensor& log_alpha,
                             int64_t BLANK,
                             bool zero_infinity) {
  using namespace mps;

  int64_t batch_size = log_probs.size(1);
  int64_t max_input_length = log_probs.size(0);

  std::vector<int64_t> target_batch_offsets(batch_size);

  int64_t pos = 0;
  int64_t tg_batch_stride = targets.stride(0);
  for (int64_t i = 0; i < batch_size; i++) {
    if (targets.dim() == 1) {
      target_batch_offsets[i] = pos;
      pos += target_lengths[i];
    } else {
      target_batch_offsets[i] = i * tg_batch_stride;
    }
  }

  // grad initialized to neginf (log-domain zero) for the scatter-logsumexp in kernel 2.
  Tensor grad = at::full_like(log_probs, -std::numeric_limits<double>::infinity(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  if (batch_size == 0 || max_input_length == 0) {
    return grad;
  }

  bool can_use_32bit_index_math = at::native::canUse32BitIndexMath(log_probs) &&
      at::native::canUse32BitIndexMath(targets) && at::native::canUse32BitIndexMath(log_alpha);
  auto metadata_dtype = can_use_32bit_index_math ? kInt : kLong;

  Tensor input_lengths_t = at::tensor(input_lengths, log_probs.options().dtype(metadata_dtype));
  Tensor target_lengths_t = at::tensor(target_lengths, log_probs.options().dtype(metadata_dtype));
  Tensor target_batch_offsets_t = at::tensor(target_batch_offsets, log_probs.options().dtype(metadata_dtype));

  if (can_use_32bit_index_math) {
    ctc_loss_backward_mps_kernel<int32_t>(grad,
                                          grad_out,
                                          log_alpha,
                                          log_probs,
                                          targets,
                                          input_lengths_t,
                                          target_lengths_t,
                                          loss,
                                          target_batch_offsets_t,
                                          BLANK,
                                          max_input_length,
                                          batch_size,
                                          zero_infinity);
  } else {
    ctc_loss_backward_mps_kernel<int64_t>(grad,
                                          grad_out,
                                          log_alpha,
                                          log_probs,
                                          targets,
                                          input_lengths_t,
                                          target_lengths_t,
                                          loss,
                                          target_batch_offsets_t,
                                          BLANK,
                                          max_input_length,
                                          batch_size,
                                          zero_infinity);
  }

  return grad;
}

} // namespace at::native
