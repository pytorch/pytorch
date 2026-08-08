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
#include <ATen/ops/_log_softmax_backward_data_native.h>
#include <ATen/ops/_log_softmax_native.h>
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

// The tiled non-last-dim kernel (one thread per column, serial axis walk) only
// wins for a SHORT axis. Above this cap the cooperative blocked kernel is used.
static constexpr int64_t kSoftmaxTiledAxisCap = 32;

// Launch plan for the blocked non-last-dim kernels.
//   cols_per_tg      : adjacent columns per threadgroup (coalesced; SIMD width)
//   tg_size          : threadgroup size = cols_per_tg * num_axis_threads
//   num_axis_chunks  : axis-split factor across threadgroups (1 => single pass)
// cols_per_tg is kept at the SIMD width (32) when inner_size allows so each warp
// load spans 32 adjacent columns; the rest of the threadgroup goes to axis
// parallelism. When the column count is small (so the single-pass grid would
// starve occupancy) and the axis is long, num_axis_chunks > 1 splits the axis
// across threadgroups via the two-pass blocked kernels.
struct BlockedLaunch {
  int64_t cols_per_tg;
  int64_t tg_size;
  int64_t num_axis_chunks;
};

static BlockedLaunch computeBlockedLaunch(int64_t inner_size, int64_t axis_size, int64_t outer_before) {
  int64_t cols_per_tg = std::min<int64_t>(inner_size, 32);
  int64_t max_axis_threads = 1024 / cols_per_tg;
  int64_t nat = 1;
  while (nat * 2 <= max_axis_threads && nat * 2 <= axis_size)
    nat *= 2;
  int64_t tg_size = cols_per_tg * nat;

  // Single-pass threadgroup count. The two-pass axis split adds a full extra
  // global round-trip (partials write + re-read), so it only pays off when the
  // single-pass grid would badly starve the GPU (very few columns). Above a
  // small TG floor the single-pass blocked kernel already fills the machine and
  // is cheaper. Empirically the split wins for the tall, few-column case
  // (65536x128 -> 4 TGs) but loses for square (1024x1024 -> 32 TGs).
  int64_t num_col_tiles = (inner_size + cols_per_tg - 1) / cols_per_tg;
  int64_t single_pass_tgs = num_col_tiles * outer_before;
  int64_t num_axis_chunks = 1;
  constexpr int64_t kMinSinglePassTGs = 24; // below this, split the axis
  constexpr int64_t kTargetTGs = 192; // aim when splitting
  constexpr int64_t kMinChunkElems = 512; // keep each chunk's work worthwhile
  if (single_pass_tgs < kMinSinglePassTGs && axis_size >= 2 * kMinChunkElems) {
    int64_t want = (kTargetTGs + single_pass_tgs - 1) / single_pass_tgs;
    int64_t max_by_work = axis_size / kMinChunkElems;
    num_axis_chunks = std::min(want, max_by_work);
    if (num_axis_chunks < 1)
      num_axis_chunks = 1;
  }
  return {cols_per_tg, tg_size, num_axis_chunks};
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

// Legacy MPSGraph log_softmax fallback (gated). Mirrors the math the previous
// log_softmax_mps_out used; retained only for the rank > 16 case the Metal
// kernels do not cover. out = (x - max) - log(sum(exp(x - max))).
static void log_softmax_mps_out_graph(const Tensor& input, int64_t dim_, const Tensor& output) {
  using CachedGraph = MPSUnaryCachedGraph;
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "log_softmax_mps_out" + getTensorsStringKey({input}) + ":" + std::to_string(dim_);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);

      MPSGraphTensor* maximumsTensor = [mpsGraph reductionMaximumWithTensor:inputTensor axis:dim_ name:nil];
      MPSGraphTensor* inputTensorSubMax = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                                 secondaryTensor:maximumsTensor
                                                                            name:nil];
      MPSGraphTensor* exponentTensor = [mpsGraph exponentWithTensor:inputTensorSubMax name:nil];
      MPSGraphTensor* exponentTensorReduced = [mpsGraph reductionSumWithTensor:exponentTensor axis:dim_ name:nil];
      MPSGraphTensor* logSumExpTensor = [mpsGraph logarithmWithTensor:exponentTensorReduced name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph subtractionWithPrimaryTensor:inputTensorSubMax
                                                            secondaryTensor:logSumExpTensor
                                                                       name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, input);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

// Legacy MPSGraph log_softmax backward fallback (gated; rank > 16 only).
// grad_input = grad_output - exp(output) * sum(grad_output).
static void log_softmax_backward_mps_out_graph(const Tensor& grad_output,
                                               const Tensor& output,
                                               int64_t dim_,
                                               const Tensor& grad_input) {
  using CachedGraph = MPSUnaryGradCachedGraph;
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "log_softmax_backward_mps_out:" + getMPSTypeString(grad_output) + ":" + std::to_string(dim_);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(grad_output));
      MPSGraphTensor* outputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(output));

      MPSGraphTensor* expTensor = [mpsGraph exponentWithTensor:outputTensor name:nil];
      MPSGraphTensor* sumTensor = [mpsGraph reductionSumWithTensor:gradOutputTensor axis:dim_ name:nil];
      MPSGraphTensor* multiplicationTensor = [mpsGraph multiplicationWithPrimaryTensor:expTensor
                                                                       secondaryTensor:sumTensor
                                                                                  name:nil];
      MPSGraphTensor* resultTensor = [mpsGraph subtractionWithPrimaryTensor:gradOutputTensor
                                                            secondaryTensor:multiplicationTensor
                                                                       name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
      newCachedGraph->gradInputTensor_ = resultTensor;
    });

    Placeholder gradPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    Placeholder resultPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);
    auto feeds = dictionaryFromPlaceholders(gradPlaceholder, outputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, resultPlaceholder);
  }
}

} // namespace mps

// Shared metal-launch implementation for softmax (is_log == false) and
// log_softmax (is_log == true). The kernel families are identical except for
// the final normalization (1/sum_exp vs subtract log(sum_exp)) and, for
// backward, the reduction (sum(dy*y) vs sum(dy)) plus the exp(out) factor; all
// of that is selected inside the kernels by an IS_LOG template bool, so here we
// only switch the kernel-name prefix ("softmax" vs "logsoftmax"). The rank > 16
// case routes to the gated MPSGraph fallback (softmax_*/log_softmax_* graph).
static void softmax_out_impl(const Tensor& input_, const int64_t dim, const Tensor& output, bool is_log) {
  using namespace mps;
  const std::string op = is_log ? "logsoftmax" : "softmax";

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
    if (is_log)
      mps::log_softmax_mps_out_graph(input, dim_, output_);
    else
      mps::softmax_mps_out_graph(input, dim_, output_);
    return;
  }

  int64_t axis_size = input.size(dim_);
  int64_t outer_size = input.numel() / axis_size;
  auto params = makeForwardParams(input, output_, dim_);

  // Tiled path: each thread does one complete softmax row at one inner position.
  // Coalesced across threads, but the per-thread axis reduction is serial, so it
  // is only a win when the axis is SHORT (otherwise the blocked path below adds
  // per-column parallelism). The win regime is large inner_size + tiny axis
  // (e.g. 8x2048x4096 dim=0, axis=8).
  {
    int64_t ndim = input.dim();
    int64_t inner_size = input.stride(dim_);
    bool use_tiled = (dim_ != ndim - 1) && input.is_contiguous() && output_.is_contiguous();
    use_tiled = use_tiled && (inner_size >= axis_size) && (axis_size <= kSoftmaxTiledAxisCap);
    if (use_tiled) {
      int64_t outer_before = outer_size / inner_size;
      int64_t tile_tg_size = std::min(inner_size, static_cast<int64_t>(1024));
      int64_t num_tiles = (inner_size + tile_tg_size - 1) / tile_tg_size;
      while (num_tiles * outer_before < 64 && tile_tg_size > 32) {
        tile_tg_size /= 2;
        num_tiles = (inner_size + tile_tg_size - 1) / tile_tg_size;
      }
      params.num_chunks = static_cast<uint32_t>(num_tiles);

      MPSStream* stream = getCurrentMPSStream();
      @autoreleasepool {
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          auto metalType = mps::scalarToMetalTypeString(input);
          auto kernel = mps::lib.getPipelineStateForFunc((op + "_forward_tiled_") + metalType);
          id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
          [encoder setComputePipelineState:kernel];
          mps::mtl_setArgs(encoder, input, output_, params);
          MTLSize threadsPerGroup = MTLSizeMake(tile_tg_size, 1, 1);
          MTLSize numGroups = MTLSizeMake(num_tiles * outer_before, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        });
      }
      return;
    }
  }

  // Blocked path: cooperative axis reduction for non-last-dim with a long axis
  // (square dim=0, tall dim=0, inner<axis). A threadgroup owns COLS_PER_TG
  // adjacent columns and packs num_axis_threads threads per column that
  // cooperate over the axis. Coalesced reads, per-column parallelism, no axis
  // cap. Supersedes the old coalesced path for these shapes.
  {
    int64_t ndim = input.dim();
    int64_t inner_size = input.stride(dim_);
    bool use_blocked = (dim_ != ndim - 1) && input.is_contiguous() && output_.is_contiguous() && (inner_size >= 1);
    if (use_blocked) {
      int64_t outer_before = outer_size / inner_size;
      auto launch = computeBlockedLaunch(inner_size, axis_size, outer_before);
      int64_t cols_per_tg = launch.cols_per_tg;
      int64_t tg_size = launch.tg_size;
      int64_t num_axis_chunks = launch.num_axis_chunks;
      params.num_chunks = static_cast<uint32_t>(cols_per_tg);
      params.num_col_chunks = static_cast<uint32_t>(num_axis_chunks);
      int64_t num_col_tiles = (inner_size + cols_per_tg - 1) / cols_per_tg;

      if (num_axis_chunks > 1) {
        // Two-pass axis split for low-column / long-axis (raises occupancy).
        Tensor blk_partials =
            at::empty({outer_before * inner_size * num_axis_chunks * 2}, input.options().dtype(at::kFloat));
        MPSStream* stream = getCurrentMPSStream();
        @autoreleasepool {
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            auto metalType = mps::scalarToMetalTypeString(input);
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            MTLSize threadsPerGroup = MTLSizeMake(tg_size, 1, 1);
            MTLSize numGroups = MTLSizeMake(num_col_tiles * outer_before * num_axis_chunks, 1, 1);

            auto reduce_kernel = mps::lib.getPipelineStateForFunc("softmax_forward_blocked2_reduce_" + metalType);
            [encoder setComputePipelineState:reduce_kernel];
            mps::mtl_setArgs(encoder, input, blk_partials, params);
            [encoder setThreadgroupMemoryLength:tg_size * 2 * sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];

            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            auto write_kernel = mps::lib.getPipelineStateForFunc((op + "_forward_blocked2_write_") + metalType);
            [encoder setComputePipelineState:write_kernel];
            mps::mtl_setArgs(encoder, input, output_, blk_partials, params);
            [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
          });
        }
        return;
      }

      MPSStream* stream = getCurrentMPSStream();
      @autoreleasepool {
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          auto metalType = mps::scalarToMetalTypeString(input);
          auto kernel = mps::lib.getPipelineStateForFunc((op + "_forward_blocked_") + metalType);
          id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
          [encoder setComputePipelineState:kernel];
          mps::mtl_setArgs(encoder, input, output_, params);
          [encoder setThreadgroupMemoryLength:tg_size * 2 * sizeof(float) atIndex:0];
          MTLSize threadsPerGroup = MTLSizeMake(tg_size, 1, 1);
          MTLSize numGroups = MTLSizeMake(num_col_tiles * outer_before, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        });
      }
      return;
    }
  }

  // Coalesced path: flat loads with shared memory reduction (legacy; the blocked
  // path above now covers the non-last-dim contiguous cases).
  {
    int64_t ndim = input.dim();
    int64_t inner_size = input.stride(dim_);
    bool use_coalesced = (dim_ != ndim - 1) && input.is_contiguous() && output_.is_contiguous() && (inner_size > 1) &&
        (inner_size < axis_size) && (axis_size <= 16384);
    if (use_coalesced) {
      int64_t outer_before = outer_size / inner_size;
      int64_t nat = 1;
      while (nat * 2 <= 1024 / inner_size)
        nat *= 2;
      int64_t coal_tg_size = inner_size * nat;
      params.num_chunks = static_cast<uint32_t>(nat);

      MPSStream* stream = getCurrentMPSStream();
      @autoreleasepool {
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          auto metalType = mps::scalarToMetalTypeString(input);
          auto kernel = mps::lib.getPipelineStateForFunc((op + "_forward_coalesced_") + metalType);
          id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
          [encoder setComputePipelineState:kernel];
          mps::mtl_setArgs(encoder, input, output_, params);
          [encoder setThreadgroupMemoryLength:coal_tg_size * 2 * sizeof(float) atIndex:0];
          MTLSize threadsPerGroup = MTLSizeMake(coal_tg_size, 1, 1);
          MTLSize numGroups = MTLSizeMake(outer_before, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        });
      }
      return;
    }
  }

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

  constexpr int64_t kFwdMinOccupancyTG = 8;
  int64_t elems_per_tg = tg_size * N_READS;
  int64_t raw_chunks = axis_size / elems_per_tg;
  int64_t max_chunks = std::min(raw_chunks, static_cast<int64_t>(16));
  bool use_two_pass_fwd = (raw_chunks >= 8) && (outer_size < kFwdMinOccupancyTG);

  Tensor fwd_partials;
  if (use_two_pass_fwd) {
    params.num_chunks = static_cast<uint32_t>(max_chunks);
    fwd_partials = at::empty({outer_size * max_chunks * 2}, input.options().dtype(at::kFloat));
  }

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      auto metalType = mps::scalarToMetalTypeString(input);
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      MTLSize threadsPerGroup = MTLSizeMake(tg_size, 1, 1);

      if (use_two_pass_fwd) {
        auto reduce_kernel = mps::lib.getPipelineStateForFunc("softmax_forward_2pass_reduce_" + metalType);
        [encoder setComputePipelineState:reduce_kernel];
        mps::mtl_setArgs(encoder, input, fwd_partials, params);
        MTLSize numGroups = MTLSizeMake(static_cast<NSUInteger>(params.num_chunks) * outer_size, 1, 1);
        [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];

        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        auto write_kernel = mps::lib.getPipelineStateForFunc((op + "_forward_2pass_write_") + metalType);
        [encoder setComputePipelineState:write_kernel];
        mps::mtl_setArgs(encoder, input, output_, fwd_partials, params);
        [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
      } else {
        id<MTLComputePipelineState> kernel;
        MTLSize srGroup = threadsPerGroup;
        if (axis_size <= 1024 * N_READS) {
          if (wide8_eligible) {
            kernel = mps::lib.getPipelineStateForFunc((op + "_forward_single_row8_") + metalType);
            srGroup = MTLSizeMake(tg_size8, 1, 1);
          } else {
            kernel = mps::lib.getPipelineStateForFunc((op + "_forward_single_row_") + metalType);
          }
        } else {
          kernel = mps::lib.getPipelineStateForFunc((op + "_forward_looped_") + metalType);
        }

        [encoder setComputePipelineState:kernel];
        mps::mtl_setArgs(encoder, input, output_, params);
        MTLSize numGroups = MTLSizeMake(outer_size, 1, 1);
        [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:srGroup];
      }
    });
  }
}

TORCH_IMPL_FUNC(softmax_mps_out)
(const Tensor& input_, const int64_t dim, const bool half_to_float, const Tensor& output) {
  TORCH_CHECK(!half_to_float, "softmax with half to float conversion is not supported on MPS");
  TORCH_CHECK(c10::isFloatingType(input_.scalar_type()), "softmax only supported for floating types");
  if (input_.numel() == 0) {
    return;
  }
  softmax_out_impl(input_, dim, output, /*is_log=*/false);
}

TORCH_IMPL_FUNC(log_softmax_mps_out)
(const Tensor& input_, const int64_t dim, const bool half_to_float, const Tensor& output) {
  // Preserve the exact BC error strings the previous MPSGraph path raised.
  TORCH_CHECK_NOT_IMPLEMENTED(input_.scalar_type() != kLong, "MPS doesn't know how to do exponent_i64");
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(input_.scalar_type()),
                              "log_softmax for complex is not supported for MPS");
  TORCH_CHECK_NOT_IMPLEMENTED(input_.scalar_type() != kBool, "log_softmax for bool is not supported for MPS");
  // The Metal kernels are instantiated only for float/half/bfloat; any other
  // (e.g. integer) dtype would miss its specialization and fail with an opaque
  // pipeline-creation error, so reject non-floating types cleanly. The kLong
  // check above keeps that type's legacy BC message and runs first.
  TORCH_CHECK(c10::isFloatingType(input_.scalar_type()), "log_softmax only supported for floating types");
  TORCH_CHECK(!half_to_float, "log_softmax with half to float conversion is not supported on MPS");
  if (input_.numel() == 0) {
    return;
  }
  softmax_out_impl(input_, dim, output, /*is_log=*/true);
}

// Shared metal-launch backward for softmax (is_log == false) and log_softmax
// (is_log == true); the IS_LOG kernel template selects the math, this only
// switches the kernel-name prefix and the MPSGraph fallback target.
static void softmax_backward_out_impl(const Tensor& grad_,
                                      const Tensor& output_,
                                      int64_t dim,
                                      const Tensor& grad_input,
                                      bool is_log) {
  using namespace mps;
  const std::string op = is_log ? "logsoftmax" : "softmax";

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
    if (is_log)
      mps::log_softmax_backward_mps_out_graph(grad, output, dim_, grad_input_);
    else
      mps::softmax_backward_mps_out_graph(grad, output, dim_, grad_input_);
    return;
  }

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

  // Tiled path for non-last-dim backward (short axis only; see fwd note).
  {
    int64_t ndim = grad.dim();
    bool use_tiled =
        (dim_ != ndim - 1) && grad.is_contiguous() && output.is_contiguous() && grad_input_.is_contiguous();
    int64_t inner_size = grad.stride(dim_);
    use_tiled = use_tiled && (inner_size >= axis_size) && (axis_size <= kSoftmaxTiledAxisCap);
    if (use_tiled) {
      int64_t outer_before = outer_size / inner_size;
      int64_t tile_tg_size = std::min(inner_size, static_cast<int64_t>(1024));
      int64_t num_tiles = (inner_size + tile_tg_size - 1) / tile_tg_size;
      while (num_tiles * outer_before < 64 && tile_tg_size > 32) {
        tile_tg_size /= 2;
        num_tiles = (inner_size + tile_tg_size - 1) / tile_tg_size;
      }
      params.num_chunks = static_cast<uint32_t>(num_tiles);

      MPSStream* stream = getCurrentMPSStream();
      @autoreleasepool {
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          auto metalType = mps::scalarToMetalTypeString(output);
          auto kernel = mps::lib.getPipelineStateForFunc((op + "_backward_tiled_") + metalType);
          id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
          [encoder setComputePipelineState:kernel];
          mps::mtl_setArgs(encoder, grad, output, grad_input_, params);
          MTLSize threadsPerGroup = MTLSizeMake(tile_tg_size, 1, 1);
          MTLSize numGroups = MTLSizeMake(num_tiles * outer_before, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        });
      }
      return;
    }
  }

  // Blocked path: cooperative axis reduction for non-last-dim with a long axis.
  {
    int64_t ndim = grad.dim();
    int64_t inner_size = grad.stride(dim_);
    bool use_blocked = (dim_ != ndim - 1) && grad.is_contiguous() && output.is_contiguous() &&
        grad_input_.is_contiguous() && (inner_size >= 1);
    if (use_blocked) {
      int64_t outer_before = outer_size / inner_size;
      auto launch = computeBlockedLaunch(inner_size, axis_size, outer_before);
      int64_t cols_per_tg = launch.cols_per_tg;
      int64_t blk_tg_size = launch.tg_size;
      int64_t num_axis_chunks = launch.num_axis_chunks;
      params.num_chunks = static_cast<uint32_t>(cols_per_tg);
      params.num_col_chunks = static_cast<uint32_t>(num_axis_chunks);
      int64_t num_col_tiles = (inner_size + cols_per_tg - 1) / cols_per_tg;

      if (num_axis_chunks > 1) {
        Tensor blk_partials =
            at::empty({outer_before * inner_size * num_axis_chunks}, grad.options().dtype(at::kFloat));
        MPSStream* stream = getCurrentMPSStream();
        @autoreleasepool {
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            auto metalType = mps::scalarToMetalTypeString(output);
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            MTLSize threadsPerGroup = MTLSizeMake(blk_tg_size, 1, 1);
            MTLSize numGroups = MTLSizeMake(num_col_tiles * outer_before * num_axis_chunks, 1, 1);

            auto dot_kernel = mps::lib.getPipelineStateForFunc((op + "_backward_blocked2_dot_") + metalType);
            [encoder setComputePipelineState:dot_kernel];
            mps::mtl_setArgs(encoder, grad, output, blk_partials, params);
            [encoder setThreadgroupMemoryLength:blk_tg_size * sizeof(float) atIndex:0];
            [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];

            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

            auto grad_kernel = mps::lib.getPipelineStateForFunc((op + "_backward_blocked2_grad_") + metalType);
            [encoder setComputePipelineState:grad_kernel];
            mps::mtl_setArgs(encoder, grad, output, grad_input_, blk_partials, params);
            [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
          });
        }
        return;
      }

      MPSStream* stream = getCurrentMPSStream();
      @autoreleasepool {
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          auto metalType = mps::scalarToMetalTypeString(output);
          auto kernel = mps::lib.getPipelineStateForFunc((op + "_backward_blocked_") + metalType);
          id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
          [encoder setComputePipelineState:kernel];
          mps::mtl_setArgs(encoder, grad, output, grad_input_, params);
          [encoder setThreadgroupMemoryLength:blk_tg_size * sizeof(float) atIndex:0];
          MTLSize threadsPerGroup = MTLSizeMake(blk_tg_size, 1, 1);
          MTLSize numGroups = MTLSizeMake(num_col_tiles * outer_before, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        });
      }
      return;
    }
  }

  // Coalesced path: flat loads with shared memory reduction (legacy).
  {
    int64_t ndim = grad.dim();
    int64_t inner_size = grad.stride(dim_);
    bool use_coalesced = (dim_ != ndim - 1) && grad.is_contiguous() && output.is_contiguous() &&
        grad_input_.is_contiguous() && (inner_size > 1) && (inner_size < axis_size) && (axis_size <= 16384);
    if (use_coalesced) {
      int64_t outer_before = outer_size / inner_size;
      int64_t nat = 1;
      while (nat * 2 <= 1024 / inner_size)
        nat *= 2;
      int64_t coal_tg_size = inner_size * nat;
      params.num_chunks = static_cast<uint32_t>(nat);

      MPSStream* stream = getCurrentMPSStream();
      @autoreleasepool {
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          auto metalType = mps::scalarToMetalTypeString(output);
          auto kernel = mps::lib.getPipelineStateForFunc((op + "_backward_coalesced_") + metalType);
          id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
          [encoder setComputePipelineState:kernel];
          mps::mtl_setArgs(encoder, grad, output, grad_input_, params);
          [encoder setThreadgroupMemoryLength:coal_tg_size * 2 * sizeof(float) atIndex:0];
          MTLSize threadsPerGroup = MTLSizeMake(coal_tg_size, 1, 1);
          MTLSize numGroups = MTLSizeMake(outer_before, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        });
      }
      return;
    }
  }

  constexpr int64_t kMinOccupancyTG = 8;
  int64_t elems_per_tg = tg_size * N_READS;
  int64_t raw_chunks = axis_size / elems_per_tg;
  int64_t max_chunks = std::min(raw_chunks, static_cast<int64_t>(16));
  bool use_two_pass = (raw_chunks >= 8) && (outer_size < kMinOccupancyTG);

  Tensor partial_sums;
  if (use_two_pass) {
    params.num_chunks = static_cast<uint32_t>(max_chunks);
    partial_sums = at::empty({outer_size * max_chunks}, grad.options().dtype(at::kFloat));
  }

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      auto metalType = mps::scalarToMetalTypeString(output);
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      MTLSize threadsPerGroup = MTLSizeMake(tg_size, 1, 1);

      if (use_two_pass) {
        auto dot_kernel = mps::lib.getPipelineStateForFunc((op + "_backward_2pass_dot_") + metalType);
        [encoder setComputePipelineState:dot_kernel];
        mps::mtl_setArgs(encoder, grad, output, partial_sums, params);
        MTLSize numGroups = MTLSizeMake(static_cast<NSUInteger>(params.num_chunks) * outer_size, 1, 1);
        [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];

        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        auto grad_kernel = mps::lib.getPipelineStateForFunc((op + "_backward_2pass_grad_") + metalType);
        [encoder setComputePipelineState:grad_kernel];
        mps::mtl_setArgs(encoder, grad, output, grad_input_, partial_sums, params);
        [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
      } else {
        id<MTLComputePipelineState> kernel;
        MTLSize srGroup = threadsPerGroup;
        if (axis_size <= 1024 * N_READS) {
          if (wide8_eligible) {
            kernel = mps::lib.getPipelineStateForFunc((op + "_backward_single_row8_") + metalType);
            srGroup = MTLSizeMake(tg_size8, 1, 1);
          } else {
            kernel = mps::lib.getPipelineStateForFunc((op + "_backward_single_row_") + metalType);
          }
        } else {
          kernel = mps::lib.getPipelineStateForFunc((op + "_backward_looped_") + metalType);
        }

        [encoder setComputePipelineState:kernel];
        mps::mtl_setArgs(encoder, grad, output, grad_input_, params);
        MTLSize numGroups = MTLSizeMake(outer_size, 1, 1);
        [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:srGroup];
      }
    });
  }
}

TORCH_IMPL_FUNC(softmax_backward_mps_out)
(const Tensor& grad_, const Tensor& output_, int64_t dim, ScalarType input_dtype, const Tensor& grad_input) {
  if (output_.numel() == 0) {
    return;
  }
  softmax_backward_out_impl(grad_, output_, dim, grad_input, /*is_log=*/false);
}

TORCH_IMPL_FUNC(log_softmax_backward_mps_out)
(const Tensor& grad_output, const Tensor& output, int64_t dim, ScalarType input_dtype, const Tensor& out) {
  // Preserve the exact BC error strings the previous MPSGraph path raised.
  TORCH_CHECK_NOT_IMPLEMENTED(grad_output.scalar_type() != kLong, "MPS doesn't know how to do exponent_i64");
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(grad_output.scalar_type()),
                              "log_softmax for complex is not supported for MPS");
  TORCH_CHECK_NOT_IMPLEMENTED(grad_output.scalar_type() != kBool, "log_softmax for bool is not supported for MPS");
  if (output.numel() == 0) {
    return;
  }
  softmax_backward_out_impl(grad_output, output, dim, out, /*is_log=*/true);
}

} // namespace at::native
