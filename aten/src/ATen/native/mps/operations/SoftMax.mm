//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/SoftMaxKernel.h>
#include <c10/util/env.h>

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_softmax_backward_data_native.h>
#include <ATen/ops/_softmax_native.h>
#endif

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/SoftMaxKernel_metallib.h>
#endif

// Gate that selects the native Metal kernels over the legacy MPSGraph path.
// The Metal kernels handle every floating-point softmax over an axis with a
// well-defined extent; the MPSGraph fallback below is retained only for the
// correctness-required cases the Metal kernels do not cover (currently the
// pre-macOS-15 ChannelsLast axis remapping). Keeping this as the normal path
// is what eliminates the per-shape MPSGraph cache growth.
static bool canUseMetalSoftmax(const Tensor& input) {
  // Escape hatch (also used by benchmarks/mps/bench_softmax.py for same-build
  // A/B vs the kept MPSGraph path): force the MPSGraph route. Lets a user opt a
  // workload back to MPSGraph without rebuilding if a shape regresses.
  static const bool force_mpsgraph = c10::utils::has_env("PYTORCH_MPS_FORCE_MPSGRAPH_SOFTMAX");
  if (force_mpsgraph) {
    return false;
  }
  // SoftmaxParams packs sizes/strides as uint32; fall back to MPSGraph for
  // extents that would overflow it (multi-billion-element tensors).
  constexpr int64_t kMaxU32 = 0xFFFFFFFFLL;
  if (input.numel() > kMaxU32) {
    return false;
  }
  for (int64_t d = 0; d < input.dim(); d++) {
    if (input.size(d) > kMaxU32 || input.stride(d) > kMaxU32) {
      return false;
    }
  }
  // SoftmaxParams holds 15 outer-dim slots (reduced dim + 15 outer = rank 16);
  // higher ranks would overflow the param block, so fall back to MPSGraph.
  return input.dim() > 0 && input.dim() <= 16;
}

static SoftmaxParams makeForwardParams(const Tensor& input, const Tensor& output, int64_t dim) {
  SoftmaxParams params = {};
  int64_t ndim = input.dim();
  params.axis_size = static_cast<uint32_t>(input.size(dim));
  params.stride_a = static_cast<uint32_t>(input.stride(dim));
  params.stride_b = static_cast<uint32_t>(output.stride(dim));
  params.ndim = static_cast<uint32_t>(ndim);
  int outer_idx = 0;
  for (int64_t d = 0; d < ndim; d++) {
    if (d == dim)
      continue;
    params.outer_sizes[outer_idx] = static_cast<uint32_t>(input.size(d));
    params.outer_strides_a[outer_idx] = static_cast<uint32_t>(input.stride(d));
    params.outer_strides_b[outer_idx] = static_cast<uint32_t>(output.stride(d));
    outer_idx++;
  }
  return params;
}

static SoftmaxParams makeBackwardParams(const Tensor& grad,
                                        const Tensor& output,
                                        const Tensor& grad_input,
                                        int64_t dim) {
  SoftmaxParams params = {};
  int64_t ndim = grad.dim();
  params.axis_size = static_cast<uint32_t>(grad.size(dim));
  params.stride_a = static_cast<uint32_t>(grad.stride(dim));
  params.stride_b = static_cast<uint32_t>(output.stride(dim));
  params.stride_c = static_cast<uint32_t>(grad_input.stride(dim));
  params.ndim = static_cast<uint32_t>(ndim);
  int outer_idx = 0;
  for (int64_t d = 0; d < ndim; d++) {
    if (d == dim)
      continue;
    params.outer_sizes[outer_idx] = static_cast<uint32_t>(grad.size(d));
    params.outer_strides_a[outer_idx] = static_cast<uint32_t>(grad.stride(d));
    params.outer_strides_b[outer_idx] = static_cast<uint32_t>(output.stride(d));
    params.outer_strides_c[outer_idx] = static_cast<uint32_t>(grad_input.stride(d));
    outer_idx++;
  }
  return params;
}

// ============================================================================
// Legacy MPSGraph fallback (gated). Retained for correctness on the cases the
// native Metal softmax does not cover. Do not delete; this is the fallback the
// canUseMetalSoftmax guard routes to.
// ============================================================================

static void get_shapes(MPSShape* input_shape_readonly,
                       NSMutableArray<NSNumber*>*& input_shape,
                       int num_input_dims,
                       c10::MemoryFormat memory_format) {
  // Modify the shape
  if (memory_format == at::MemoryFormat::Contiguous) {
    for (int i = 0; i < num_input_dims; i++)
      input_shape[i] = input_shape_readonly[i];
  } else { // ChannelsLast
    auto num_channels = input_shape_readonly[1];
    input_shape[0] = input_shape_readonly[0];
    for (int i = 1; i < num_input_dims - 1; i++)
      input_shape[i] = input_shape_readonly[i + 1];
    input_shape[num_input_dims - 1] = num_channels;
  }
}

static void softmax_mps_out_graph(const Tensor& input, int64_t dim_, const Tensor& output) {
  static const bool is_macOS_15_0_or_newer = is_macos_at_least(MacOSVersion::MACOS_15_0);
  const auto memory_format = input.suggest_memory_format();

  using CachedGraph = MPSUnaryCachedGraph;
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string mem_format_key = get_mem_format_string(memory_format);
    MPSShape* input_shape_readonly = mps::getMPSShape(input);
    int num_input_dims = [input_shape_readonly count];
    // Check - Channels last implies 4d
    TORCH_CHECK(memory_format != at::MemoryFormat::ChannelsLast || num_input_dims == 4,
                "ChannelsLast implies 4d tensor")
    // Input shape changes based on memory format
    NSMutableArray<NSNumber*>* input_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];

    get_shapes(input_shape_readonly, input_shape, num_input_dims, memory_format);

    // Change dim
    if (memory_format == at::MemoryFormat::ChannelsLast && dim_ > 0 && !is_macOS_15_0_or_newer) {
      switch (dim_) {
        case 1:
          dim_ = 3;
          break;
        case 2:
          dim_ = 1;
          break;
        case 3:
          dim_ = 2;
          break;
        default:
          assert(0 && "Invalid dim\n");
      }
    }

    std::string key = "softmax_mps_out" + getTensorsStringKey(input, true, /*exclude_shape*/ true) + ":" +
        mem_format_key + ":" + std::to_string(dim_);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()));

      // passing selector of softMaxWithTensor on the mpsGraph object
      MPSGraphTensor* outputTensor = [mpsGraph softMaxWithTensor:inputTensor axis:(NSInteger)dim_ name:nil];

      // Output needs to be contiguous format
      if (memory_format == at::MemoryFormat::ChannelsLast && !is_macOS_15_0_or_newer) {
        auto N = input_shape[0];
        auto H = input_shape[1];
        auto W = input_shape[2];
        auto C = input_shape[3];

        outputTensor = [mpsGraph reshapeTensor:outputTensor
                                     withShape:@[ N, ([NSNumber numberWithInt:[H intValue] * [W intValue]]), C ]
                                          name:nil];
        outputTensor = [mpsGraph transposeTensor:outputTensor dimension:1 withDimension:2 name:nil];
        outputTensor = [mpsGraph reshapeTensor:outputTensor withShape:@[ N, C, H, W ] name:nil];
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder inputPlaceholder =
        Placeholder(cachedGraph->inputTensor_, input, is_macOS_15_0_or_newer ? nil : input_shape);
    // This must be the Contiguous shape
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

static void softmax_backward_mps_out_graph(const Tensor& grad,
                                           const Tensor& output,
                                           int64_t dim_,
                                           const Tensor& grad_input) {
  using CachedGraph = MPSUnaryGradCachedGraph;
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    MPSShape* grad_shape = mps::getMPSShape(grad);
    NSString* ns_shape_key = [[grad_shape valueForKey:@"description"] componentsJoinedByString:@","];

    std::string key = "softmax_backward_mps_out:" + getMPSTypeString(output) + ":" + [ns_shape_key UTF8String] + ":" +
        std::to_string(dim_);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* softmaxTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(output), grad_shape);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad), grad_shape);

      MPSGraphTensor* mulTensor = [mpsGraph multiplicationWithPrimaryTensor:softmaxTensor
                                                            secondaryTensor:gradOutputTensor
                                                                       name:nil];
      MPSGraphTensor* mulSumTensor = [mpsGraph reductionSumWithTensor:mulTensor axis:(NSInteger)dim_ name:nil];
      MPSGraphTensor* gradSubTensor = [mpsGraph subtractionWithPrimaryTensor:gradOutputTensor
                                                             secondaryTensor:mulSumTensor
                                                                        name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:softmaxTensor
                                                                  secondaryTensor:gradSubTensor
                                                                             name:nil];

      newCachedGraph->outputTensor_ = softmaxTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    Placeholder softmaxPlaceholder = Placeholder(cachedGraph->outputTensor_, output, grad_shape);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad, grad_shape);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    auto feeds = dictionaryFromPlaceholders(softmaxPlaceholder, gradOutputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
}

} // namespace mps

TORCH_IMPL_FUNC(softmax_mps_out)
(const Tensor& input_, const int64_t dim, const bool half_to_float, const Tensor& output) {
  TORCH_CHECK(!half_to_float, "softmax with half to float conversion is not supported on MPS");
  TORCH_CHECK(c10::isFloatingType(input_.scalar_type()), "softmax only supported for floating types");

  if (input_.numel() == 0) {
    return;
  }

  Tensor input;
  Tensor output_ = output;
  if (input_.dim() == 0) {
    input = input_.view(1);
    // The structured kernel allocates output with the same (scalar) shape as
    // the input; view it as 1-D so the params/kernel index a valid axis. The
    // view shares storage, so writes land in the real 0-D output.
    output_ = output.view(1);
  } else
    input = input_;

  int64_t dim_ = maybe_wrap_dim(dim, input.dim());
  TORCH_CHECK(dim_ >= 0 && dim_ < input.dim(), "Softmax:dim must be non-negative and less than input dimensions");

  if (!mps::canUseMetalSoftmax(input)) {
    mps::softmax_mps_out_graph(input, dim_, output_);
    return;
  }
  // Last-dim softmax only in this PR: non-last-dim stays on the MPSGraph path
  // until the native non-last-dim Metal kernels land in a follow-up.
  if (dim_ != input.dim() - 1) {
    mps::softmax_mps_out_graph(input, dim_, output_);
    return;
  }

  using namespace mps;
  int64_t axis_size = input.size(dim_);
  int64_t outer_size = input.numel() / axis_size;
  auto params = makeForwardParams(input, output_, dim_);

  constexpr int N_READS = 4;
  int64_t tg_size = std::min(static_cast<int64_t>((axis_size + N_READS - 1) / N_READS), static_cast<int64_t>(1024));

  // 8-wide single-row variant for half-precision last-dim rows: each thread
  // handles 8 elements (two vec4 loads), halving the threadgroup so the per-row
  // threadgroup reduction is cheaper. Only for contiguous last-dim half/bfloat
  // rows that fit the single-row register budget (axis <= 1024 * 8).
  bool is_half = (input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16);
  bool wide8_eligible = is_half && (dim_ == input.dim() - 1) && (params.stride_a == 1) && (params.stride_b == 1) &&
      (axis_size <= 1024 * 8);
  int64_t tg_size8 = std::min(static_cast<int64_t>((axis_size + 8 - 1) / 8), static_cast<int64_t>(1024));

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      auto metalType = mps::scalarToMetalTypeString(input);
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      MTLSize threadsPerGroup = MTLSizeMake(tg_size, 1, 1);
      id<MTLComputePipelineState> kernel;
      MTLSize srGroup = threadsPerGroup;
      if (axis_size <= 1024 * N_READS) {
        if (wide8_eligible) {
          kernel = mps::lib.getPipelineStateForFunc("softmax_forward_single_row8_" + metalType);
          srGroup = MTLSizeMake(tg_size8, 1, 1);
        } else {
          kernel = mps::lib.getPipelineStateForFunc("softmax_forward_single_row_" + metalType);
        }
      } else {
        kernel = mps::lib.getPipelineStateForFunc("softmax_forward_looped_" + metalType);
      }

      [encoder setComputePipelineState:kernel];
      mps::mtl_setArgs(encoder, input, output_, params);
      MTLSize numGroups = MTLSizeMake(outer_size, 1, 1);
      [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:srGroup];
    });
  }
}

TORCH_IMPL_FUNC(softmax_backward_mps_out)
(const Tensor& grad_, const Tensor& output_, int64_t dim, ScalarType input_dtype, const Tensor& grad_input) {
  if (output_.numel() == 0) {
    return;
  }

  Tensor grad;
  if (grad_.dim() == 0) {
    grad = grad_.view(1);
  } else
    grad = grad_;

  Tensor output;
  if (output_.dim() == 0) {
    output = output_.view(1);
  } else
    output = output_;

  // The structured kernel allocates grad_input with the same (scalar) shape as
  // the inputs; view it as 1-D so the params/kernel index a valid axis. The
  // view shares storage, so writes land in the real 0-D grad_input.
  Tensor grad_input_ = grad_input;
  if (grad_input.dim() == 0) {
    grad_input_ = grad_input.view(1);
  }

  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());
  TORCH_CHECK(dim_ >= 0 && dim_ < grad.dim(), "Grad:dim must be non-negative and less than input dimensions");

  // The Metal backward kernel is specialized on a single element type; a mixed
  // half/float grad_input (e.g. softmax(x_fp16, dtype=fp32).backward()) would
  // write the wrong element size, so route dtype-mismatched cases to MPSGraph.
  bool bwd_dtypes_match =
      grad.scalar_type() == output.scalar_type() && grad_input_.scalar_type() == output.scalar_type();
  if (!(bwd_dtypes_match && mps::canUseMetalSoftmax(output) && mps::canUseMetalSoftmax(grad))) {
    mps::softmax_backward_mps_out_graph(grad, output, dim_, grad_input_);
    return;
  }
  // Last-dim softmax only in this PR (mirrors the forward gate).
  if (dim_ != grad.dim() - 1) {
    mps::softmax_backward_mps_out_graph(grad, output, dim_, grad_input_);
    return;
  }

  using namespace mps;
  int64_t axis_size = output.size(dim_);
  int64_t outer_size = output.numel() / axis_size;

  constexpr int N_READS = 4;
  int64_t tg_size = std::min(static_cast<int64_t>((axis_size + N_READS - 1) / N_READS), static_cast<int64_t>(1024));
  auto params = makeBackwardParams(grad, output, grad_input_, dim_);

  // 8-wide single-row variant for half-precision last-dim rows: each thread
  // handles 8 elements (two vec4 loads), halving the threadgroup so the per-row
  // threadgroup reduction is cheaper. Mirrors the forward 8-wide path so that
  // last-dim half fwdbwd does not lose the forward speedup to a narrow backward
  // pass. Only for contiguous last-dim half/bfloat rows that fit the single-row
  // register budget (axis <= 1024 * 8).
  bool is_half = (output.scalar_type() == at::kHalf || output.scalar_type() == at::kBFloat16);
  bool wide8_eligible = is_half && (dim_ == grad.dim() - 1) && (params.stride_a == 1) && (params.stride_b == 1) &&
      (params.stride_c == 1) && (axis_size <= 1024 * 8);
  int64_t tg_size8 = std::min(static_cast<int64_t>((axis_size + 8 - 1) / 8), static_cast<int64_t>(1024));

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      auto metalType = mps::scalarToMetalTypeString(output);
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      MTLSize threadsPerGroup = MTLSizeMake(tg_size, 1, 1);
      id<MTLComputePipelineState> kernel;
      MTLSize srGroup = threadsPerGroup;
      if (axis_size <= 1024 * N_READS) {
        if (wide8_eligible) {
          kernel = mps::lib.getPipelineStateForFunc("softmax_backward_single_row8_" + metalType);
          srGroup = MTLSizeMake(tg_size8, 1, 1);
        } else {
          kernel = mps::lib.getPipelineStateForFunc("softmax_backward_single_row_" + metalType);
        }
      } else {
        kernel = mps::lib.getPipelineStateForFunc("softmax_backward_looped_" + metalType);
      }

      [encoder setComputePipelineState:kernel];
      mps::mtl_setArgs(encoder, grad, output, grad_input_, params);
      MTLSize numGroups = MTLSizeMake(outer_size, 1, 1);
      [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:srGroup];
    });
  }
}

} // namespace at::native
