//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/Pool.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/ReduceOps.h>
#include <c10/util/irange.h>
#include <algorithm>
#include <numeric>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/all_native.h>
#include <ATen/ops/amax.h>
#include <ATen/ops/amax_native.h>
#include <ATen/ops/amin.h>
#include <ATen/ops/amin_native.h>
#include <ATen/ops/any_native.h>
#include <ATen/ops/argmax_native.h>
#include <ATen/ops/argmin_native.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/count_nonzero_native.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/max_native.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mean_native.h>
#include <ATen/ops/min_native.h>
#include <ATen/ops/nansum_native.h>
#include <ATen/ops/prod_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/std_mean_native.h>
#include <ATen/ops/std_native.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/sum_native.h>
#include <ATen/ops/trace_native.h>
#include <ATen/ops/var.h>
#include <ATen/ops/var_mean_native.h>
#include <ATen/ops/var_native.h>
#endif

namespace at::native {
using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ReduceOps_metallib.h>
#endif

enum StdVarType { STANDARD_VARIANCE, STANDARD_DEVIATION };

enum MPSReductionType {
  MAX,
  MIN,
  PROD,
  MEAN,
};

static void set_apparent_shapes(NSMutableArray<NSNumber*>*& apparent_out_shape,
                                NSMutableArray<NSNumber*>*& apparent_in_shape,
                                int64_t num_reduce_dims,
                                int64_t num_output_dims,
                                const IntArrayRef& input_shape,
                                NSMutableArray<NSNumber*>*& axes) {
  if (num_reduce_dims == 0) {
    /* Output shape becomes a one
     * Input shape becomes flattened
     * Because 0 reduce dims means all dims are reduced
     */
    apparent_in_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    int64_t num_in_elements = c10::multiply_integers(input_shape);
    apparent_in_shape[0] = [NSNumber numberWithInt:num_in_elements];

    apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    apparent_out_shape[0] = @1;
  } else {
    // num_output_dims in this case is number of input dims
    apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_output_dims];
    for (const auto i : c10::irange(num_output_dims)) {
      int64_t current_input_dim = input_shape[i];

      // If the current dim is to be reduced
      bool is_reduce_dim = false;

      for (const auto j : c10::irange(num_reduce_dims)) {
        if (i == [axes[j] intValue]) {
          is_reduce_dim = true;
          break;
        }
      }

      apparent_out_shape[i] = is_reduce_dim ? @1 : [NSNumber numberWithInt:current_input_dim];
    }
  }
}

// Helper function to set the axes of reduction
static void set_axes(NSMutableArray<NSNumber*>*& axes,
                     int64_t num_reduce_dims,
                     OptionalIntArrayRef opt_dim,
                     int64_t num_input_dims) {
  if (num_reduce_dims == 0) {
    axes = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    axes[0] = @0;
  } else {
    TORCH_INTERNAL_ASSERT(opt_dim.has_value());
    IntArrayRef dim = opt_dim.value();
    axes = [NSMutableArray<NSNumber*> arrayWithCapacity:num_reduce_dims];
    for (const auto i : c10::irange(num_reduce_dims)) {
      axes[i] = [NSNumber numberWithInt:maybe_wrap_dim(dim[i], num_input_dims)];
    }
  }
}

// Helper function to prepare axes and tensor shapes
static void set_axes_and_shapes(const IntArrayRef& input_shape,
                                OptionalIntArrayRef opt_dims,
                                NSMutableArray<NSNumber*>*& axes,
                                NSMutableArray<NSNumber*>*& apparent_input_shape,
                                NSMutableArray<NSNumber*>*& apparent_output_shape,
                                NSMutableArray<NSNumber*>*& output_shape) {
  int64_t num_input_dims = input_shape.size();
  int64_t num_reduce_dims = opt_dims.has_value() ? opt_dims.value().size() : 0;
  int64_t num_output_dims;

  num_output_dims = num_reduce_dims == 0 ? 1 : num_input_dims;

  // Reduction axes
  set_axes(axes, num_reduce_dims, opt_dims, input_shape.size());

  // Shapes
  set_apparent_shapes(apparent_output_shape, apparent_input_shape, num_reduce_dims, num_output_dims, input_shape, axes);

  // Squeeze dims for output shape
  output_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:0];
  for (const auto i : c10::irange(num_output_dims)) {
    if ([apparent_output_shape[i] longValue] != 1) {
      [output_shape addObject:apparent_output_shape[i]];
    }
  }
}

static void reduction_out_mps(const Tensor& input_t,
                              OptionalIntArrayRef opt_dim,
                              bool keepdim,
                              std::optional<ScalarType> dtype,
                              const Tensor& output_t,
                              MPSReductionType reduction_type,
                              const std::string& func_name) {
  // NS: TODO: get rid of all those shenanigans and just call reduction_op with view tensor
  bool canSqueezeLastDim = true;
  IntArrayRef input_shape = input_t.sizes();
  if (opt_dim.has_value()) {
    IntArrayRef dim = opt_dim.value();
    for (const auto dim_val : dim) {
      auto wrap_dim = maybe_wrap_dim(dim_val, input_shape.size());
      // canSqueeze logic is broken when dim is negative, it introduces off-by-one-errors or crashes
      // See https://github.com/pytorch/pytorch/issues/136132#issuecomment-2354482608
      if (wrap_dim >= 4 || dim_val < 0) {
        canSqueezeLastDim = false;
      }
      TORCH_CHECK(
          wrap_dim < static_cast<decltype(wrap_dim)>(input_shape.size() == 0 ? input_t.numel() : input_shape.size()),
          func_name + ": reduction dim must be in the range of input shape")
    }
  }

  if (input_shape.size() >= 5 && canSqueezeLastDim) {
    for (const auto i : c10::irange(4, input_shape.size())) {
      if (input_shape[i] != 1) {
        canSqueezeLastDim = false;
      }
    }
  } else {
    canSqueezeLastDim = false;
  }

  MPSShape* mpsShape = getMPSShape(input_t);
  if (canSqueezeLastDim) {
    mpsShape = @[ @(input_shape[0]), @(input_shape[1]), @(input_shape[2]), @(input_shape[3]) ];
    input_shape = makeArrayRef(input_shape.begin(), input_shape.end() - (input_t.dim() - 4));
  }

  NSMutableArray<NSNumber*>* axes = nil;
  NSMutableArray<NSNumber*>* apparent_input_shape = nil;
  NSMutableArray<NSNumber*>* apparent_output_shape = nil;
  NSMutableArray<NSNumber*>* output_shape = nil;

  set_axes_and_shapes(input_shape, opt_dim, axes, apparent_input_shape, apparent_output_shape, output_shape);
  NSArray<NSNumber*>* wrappedAxes = getTensorAxes(input_shape, opt_dim);

  if (output_t.numel() == 0 || input_t.numel() == 0) {
    switch (reduction_type) {
      case MPSReductionType::PROD:
        output_t.fill_(1);
        break;
      case MPSReductionType::MEAN:
        output_t.fill_(std::numeric_limits<float>::quiet_NaN());
        break;
      case MPSReductionType::MAX:
      case MPSReductionType::MIN:
        TORCH_CHECK(opt_dim.has_value(), "Expected reduction dim to be specified for input.numel() == 0");
        break;
      default:
        TORCH_INTERNAL_ASSERT(false, "Unexpected reduction type ", reduction_type);
        break;
    }
    return;
  }
  auto stream = getCurrentMPSStream();
  @autoreleasepool {
    std::string dtype_str = dtype.has_value() ? getMPSTypeString(dtype.value()) : "";
    NSString* ns_key = [[wrappedAxes valueForKey:@"description"] componentsJoinedByString:@","];
    std::string key = func_name + ":" + std::string([ns_key UTF8String]) + ":" + getTensorsStringKey(input_t) + ":" +
        std::to_string(keepdim) + ":" + std::to_string(reduction_type) + ":" + getTensorsStringKey(output_t) + ":" +
        dtype_str;
    using CachedGraph = MPSUnaryCachedGraph;
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputScalarType = input_t.scalar_type();

      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input_t), mpsShape);
      MPSGraphTensor* castInputTensor = inputTensor;
      MPSDataType inputCastType = MPSDataTypeInvalid;
      if (dtype.has_value() &&
          (dtype.value() == kFloat || dtype.value() == kHalf || dtype.value() == kInt || dtype.value() == kLong)) {
        inputCastType = getMPSDataType(dtype.value());
      } else if (inputScalarType != kInt && inputScalarType != kHalf && inputScalarType != kFloat &&
                 inputScalarType != kComplexFloat && inputScalarType != kComplexHalf && inputScalarType != kLong) {
        inputCastType = getMPSDataType(kFloat);
      }

      if (inputCastType != MPSDataTypeInvalid) {
        castInputTensor = castMPSTensor(mpsGraph, inputTensor, inputCastType);
      }

      MPSGraphTensor* castOutputTensor = nil;

      if (reduction_type == MPSReductionType::PROD) {
        castOutputTensor = [mpsGraph reductionProductWithTensor:castInputTensor axes:wrappedAxes name:nil];
      } else if (reduction_type == MPSReductionType::MEAN) {
        castOutputTensor = [mpsGraph meanOfTensor:castInputTensor axes:wrappedAxes name:nil];
      }

      MPSGraphTensor* outputTensor = castOutputTensor;
      if (getMPSDataType(output_t) != [castOutputTensor dataType]) {
        outputTensor = castMPSTensor(mpsGraph, castOutputTensor, output_t.scalar_type());
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t, mpsShape);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t, apparent_output_shape);
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

static void norm_kernel_mps(TensorIterator& iter, const Scalar& p_scalar) {
  const Tensor& output = iter.output(0);
  const Tensor& input = iter.input(0);
  auto p = p_scalar.to<double>();

  if (input.numel() == 0) {
    output.fill_((p < 0) ? INFINITY : 0);
    return;
  }

  if (output.numel() == 0) {
    return;
  }

  // Number of input elements that are reduced into one output element
  uint32_t reduction_size = input.numel() / output.numel();

  TORCH_INTERNAL_ASSERT(output.dim() == input.dim());

  // Fast path: L1/L2 norm over the innermost contiguous dim reuses the sum
  // inner kernel (abs/square load + sqrt)
  if ((p == 1.0 || p == 2.0) && output.numel() > 1 && input.is_contiguous() && output.is_contiguous() &&
      input.scalar_type() == output.scalar_type() &&
      (input.scalar_type() == kFloat || input.scalar_type() == kHalf || input.scalar_type() == kBFloat16)) {
    int num_reduced = 0;
    int reduced_dim = -1;
    for (const auto d : c10::irange(input.dim())) {
      if (input.size(d) != output.size(d)) {
        num_reduced++;
        reduced_dim = d;
      }
    }
    if (num_reduced == 1 && reduced_dim == input.dim() - 1) {
      uint32_t N = input.size(input.dim() - 1);
      uint32_t M = input.numel() / N;
      auto kernel_name = fmt::format("norm_{}_reduction_inner_{}_{}",
                                     p == 2.0 ? "l2" : "l1",
                                     scalarToMetalTypeString(input),
                                     scalarToMetalTypeString(output));
      constexpr uint32_t TG_SIZE = 256;
      constexpr uint32_t rows_per_tg = TG_SIZE / 32;
      const auto num_tgs = c10::metal::ceil_div(M, rows_per_tg);
      MPSStream* stream = getCurrentMPSStream();
      return dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          id<MTLComputeCommandEncoder> ce = stream->commandEncoder();
          auto ps = lib.getPipelineStateForFunc(kernel_name);
          getMPSProfiler().beginProfileKernel(ps, "norm_reduction_inner", {input});
          [ce setComputePipelineState:ps];
          mtl_setArgs(ce, input, output, std::array<uint32_t, 2>{M, N}, 0.0f);
          [ce dispatchThreads:MTLSizeMake(num_tgs * TG_SIZE, 1, 1) threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
          getMPSProfiler().endProfileKernel(ps);
        }
      });
    }
  }

  NormParams params;

  params.ndim = input.dim();
  params.p = static_cast<float>(p);
  params.reduction_size = reduction_size;

  for (const auto dim_idx : c10::irange(input.dim())) {
    params.input_sizes[dim_idx] = input.size(dim_idx);
    params.input_strides[dim_idx] = input.stride(dim_idx);
    params.output_sizes[dim_idx] = output.size(dim_idx);
    params.output_strides[dim_idx] = output.stride(dim_idx);
  }

  MPSStream* stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();
      auto pipeline_state = lib.getPipelineStateForFunc(
          fmt::format("norm_{}_{}", scalarToMetalTypeString(input), scalarToMetalTypeString(output)));
      getMPSProfiler().beginProfileKernel(pipeline_state, "norm", {input});
      [compute_encoder setComputePipelineState:pipeline_state];
      mtl_setArgs(compute_encoder, input, output, params);

      auto threads_per_group = std::min(MAX_THREADGROUP_SIZE, reduction_size);
      uint32_t num_threads = output.numel() * threads_per_group;

      [compute_encoder dispatchThreads:MTLSizeMake(num_threads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];

      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });
}

// ============================================================================
// Metal kernel dispatch helpers for welford
// ============================================================================

struct WelfordConfig {
  float denom; // host-computed max(reduction_count - correction, 0); see kernel
  float compute_std;
  float write_mean;
};

static std::vector<int64_t> get_reduce_dims(const Tensor& input, OptionalIntArrayRef opt_dim) {
  std::vector<int64_t> dims;
  if (opt_dim.has_value() && !opt_dim.value().empty()) {
    // Raises on duplicate dims, matching CPU ("dim N appears multiple times").
    at::dim_list_to_bitset(opt_dim.value(), input.dim());
    for (auto d : opt_dim.value()) {
      dims.push_back(maybe_wrap_dim(d, input.dim()));
    }
  } else {
    for (int64_t d = 0; d < input.dim(); d++) {
      dims.push_back(d);
    }
  }
  return dims;
}

static NormParams<> build_reduce_params(const Tensor& input,
                                        const std::vector<int64_t>& reduce_dims,
                                        const Tensor& output,
                                        bool keepdim) {
  TORCH_CHECK(static_cast<uint32_t>(input.dim()) <= c10::metal::max_ndim,
              "MPS var/std supports tensors with at most ",
              c10::metal::max_ndim,
              " dimensions, but got ",
              input.dim());
  NormParams params;
  params.ndim = input.dim();
  params.p = 0;
  constexpr int64_t kU32Max = std::numeric_limits<uint32_t>::max();
  auto checked_u32 = [&](int64_t value, const char* field) -> uint32_t {
    TORCH_CHECK(value >= 0 && value <= kU32Max, "MPS var/std ", field, " exceeds the uint32 indexing domain");
    return static_cast<uint32_t>(value);
  };
  params.reduction_size = checked_u32(input.numel() / std::max<int64_t>(output.numel(), 1), "reduction size");

  bool is_reduced[c10::metal::max_ndim] = {};
  for (auto d : reduce_dims)
    is_reduced[d] = true;

  if (keepdim || output.dim() == input.dim()) {
    for (uint32_t d = 0; d < params.ndim; d++) {
      params.input_sizes[d] = checked_u32(input.size(d), "input size");
      params.input_strides[d] = checked_u32(input.stride(d), "input stride");
      params.output_sizes[d] = checked_u32(output.size(d), "output size");
      params.output_strides[d] = checked_u32(output.stride(d), "output stride");
    }
  } else {
    uint32_t out_d = 0;
    for (uint32_t d = 0; d < params.ndim; d++) {
      params.input_sizes[d] = checked_u32(input.size(d), "input size");
      params.input_strides[d] = checked_u32(input.stride(d), "input stride");
      if (is_reduced[d]) {
        params.output_sizes[d] = 1;
        params.output_strides[d] = 0;
      } else {
        params.output_sizes[d] = checked_u32(output.size(out_d), "output size");
        params.output_strides[d] = checked_u32(output.stride(out_d), "output stride");
        out_d++;
      }
    }
  }

  return params;
}

static void welford_kernel_mps(const Tensor& input,
                               const std::vector<int64_t>& reduce_dims,
                               bool keepdim,
                               double correction_value,
                               bool compute_std,
                               const Tensor& output,
                               const Tensor* output_mean = nullptr) {
  if (input.numel() == 0 || output.numel() == 0)
    return;

  auto in_str = scalarToMetalTypeString(input);
  auto out_str = scalarToMetalTypeString(output);
  int64_t reduction_size = input.numel() / output.numel();
  constexpr int64_t kU32Max = std::numeric_limits<uint32_t>::max();
  TORCH_CHECK(input.numel() <= kU32Max, "MPS var/std reduction with more than 2^32 input elements is not supported");
  TORCH_CHECK(output.numel() <= kU32Max, "MPS var/std reduction with more than 2^32 output elements is not supported");
  TORCH_CHECK(reduction_size <= kU32Max, "MPS var/std reduction over more than 2^32 elements is not supported");
  const auto reduction_size_u32 = static_cast<uint32_t>(reduction_size);

  WelfordConfig config;
  // Compute the divisor on the host in double so it stays exact for large
  // reductions: a float element count loses integer precision above 2^24, which
  // corrupted denom when correction was close to N (var returned inf).
  config.denom = static_cast<float>(std::max(static_cast<double>(reduction_size) - correction_value, 0.0));
  config.compute_std = compute_std ? 1.0f : 0.0f;
  config.write_mean = output_mean ? 1.0f : 0.0f;

  Tensor mean_placeholder;
  const Tensor& mean_tensor = output_mean ? *output_mean : (mean_placeholder = at::empty({1}, output.options()));

  MPSStream* stream = getCurrentMPSStream();

  // 2-pass for all-reduce (single-output) when N is large enough to give
  // multiple TGs useful work. Single-pass welford is bottlenecked by the
  // 1024-thread single-TG limit for any-shape reduce.
  if (output.numel() == 1 && input.is_contiguous() && input.numel() > MAX_THREADGROUP_SIZE * 4) {
    uint32_t total_N = static_cast<uint32_t>(input.numel());
    uint32_t num_groups = std::min<uint32_t>(512, c10::metal::ceil_div(total_N, MAX_THREADGROUP_SIZE * 8u));
    while (num_groups > 1 && total_N % num_groups != 0) {
      num_groups--;
    }
    uint32_t elems_per_group = total_N / num_groups;

    // 4 floats/group, not 3: the pass1/pass2 kernels index partials as
    // device float3*, which Metal gives a 16-byte (4-float) array stride.
    // Allocating only 3 floats/group overruns the buffer in pass1's last group.
    auto partials = at::empty({static_cast<int64_t>(num_groups) * 4}, input.options().dtype(at::kFloat));

    auto kernel_p1 = fmt::format("welford_pass1_{}", in_str);
    auto kernel_p2 = fmt::format("welford_pass2_{}", out_str);

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

        auto ps1 = lib.getPipelineStateForFunc(kernel_p1);
        getMPSProfiler().beginProfileKernel(ps1, "welford_reduction_pass1", {input});
        [enc setComputePipelineState:ps1];
        struct {
          uint32_t elems_per_group, total_N;
        } sizes_p1 = {elems_per_group, total_N};
        mtl_setArgs(enc, input, partials, sizes_p1);
        // Round to a full simdgroup: both passes call simd_welford_combine,
        // whose simd_shuffle_and_fill_down reads undefined data from inactive
        // lanes in a partial simdgroup (0 on M1/M2 -> corrupt combine). Padding
        // threads carry the welford identity (count 0), so over-dispatching is
        // safe.
        auto tpg1 = std::min<uint32_t>(MAX_THREADGROUP_SIZE, c10::metal::ceil_div(elems_per_group, 32u) * 32u);
        [enc dispatchThreads:MTLSizeMake(num_groups * tpg1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg1, 1, 1)];
        getMPSProfiler().endProfileKernel(ps1);

        auto ps2 = lib.getPipelineStateForFunc(kernel_p2);
        getMPSProfiler().beginProfileKernel(ps2, "welford_reduction_pass2", {partials});
        [enc setComputePipelineState:ps2];
        mtl_setArgs(enc, partials, output, mean_tensor, num_groups, config);
        auto tpg2 = std::min(MAX_THREADGROUP_SIZE, c10::metal::ceil_div(num_groups, 32u) * 32u);
        [enc dispatchThreads:MTLSizeMake(tpg2, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg2, 1, 1)];
        getMPSProfiler().endProfileKernel(ps2);
      }
    });
    return;
  }

  // is_outer / is_inner are designed for shapes where one of M/N is large.
  // For output.numel() == 1 (all-reduce) either pass goes through the 2-pass
  // block above; if 2-pass didn't fire (N below its threshold), fall through
  // to the generic kernel which uses up to 1024 threads in one TG.
  bool is_single = reduce_dims.size() == 1 && output.numel() != 1;
  bool is_outer = is_single && reduce_dims[0] == 0 && input.is_contiguous() && output.is_contiguous();
  bool is_inner = is_single && reduce_dims[0] == input.dim() - 1 && input.is_contiguous() && output.is_contiguous();

  if (is_outer) {
    uint32_t M = input.size(0);
    uint32_t N = input.numel() / M;
    uint32_t TG_X, TG_Y;
    std::string kernel;
    if (M <= 8) {
      TG_X = 128;
      TG_Y = 8;
      kernel = fmt::format("welford_outer_8_{}_{}", in_str, out_str);
    } else if (M <= 16) {
      TG_X = 64;
      TG_Y = 16;
      kernel = fmt::format("welford_outer_16_{}_{}", in_str, out_str);
    } else {
      TG_X = 32;
      TG_Y = 32;
      kernel = fmt::format("welford_outer_{}_{}", in_str, out_str);
    }
    auto num_tg_x = c10::metal::ceil_div(N, TG_X);

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto ps = lib.getPipelineStateForFunc(kernel);
        getMPSProfiler().beginProfileKernel(ps, "welford_outer", {input});
        [enc setComputePipelineState:ps];
        // Pad to 4 uints: welford_reduction_outer takes `constant uint3&`,
        // which Metal sizes at 16 bytes; a 12-byte struct fails validation.
        const std::array<uint32_t, 4> sizes_s{M, N, 1, 0};
        mtl_setArgs(enc, input, output, mean_tensor, sizes_s, config);
        [enc dispatchThreads:MTLSizeMake(num_tg_x * TG_X, TG_Y, 1) threadsPerThreadgroup:MTLSizeMake(TG_X, TG_Y, 1)];
        getMPSProfiler().endProfileKernel(ps);
      }
    });
    return;
  }

  if (is_inner) {
    uint32_t N = input.size(input.dim() - 1);
    uint32_t M = input.numel() / N;
    // Tall-thin (small N, large M) -> one thread per row: a 32-lane SIMD group
    // on N<=32 leaves most lanes idle and pays simd-reduction overhead for under
    // one element of work per lane. Many short-but-not-tiny rows -> one SIMD
    // group per row (8 rows/TG). Few/huge rows (small M, large N) -> the "wide"
    // kernel (one whole TG per row, up to 1024 threads): 32 lanes on a giant row
    // is both slow and loses precision from a deep per-lane streaming Welford
    // (e.g. GroupNorm's M=2, N~19M).
    bool use_thin = (M >= 64 && N <= 32);
    bool use_simd_per_row = (M >= 64 && N <= 16384);
    uint32_t tg_size, num_tgs;
    std::string kernel;
    if (use_thin) {
      kernel = fmt::format("welford_inner_thin_{}_{}", in_str, out_str);
      tg_size = 256;
      num_tgs = c10::metal::ceil_div(M, tg_size); // one thread per row
    } else if (use_simd_per_row) {
      kernel = fmt::format("welford_inner_{}_{}", in_str, out_str);
      tg_size = 256; // 8 SIMD groups = 8 rows per TG
      num_tgs = c10::metal::ceil_div(M, tg_size / 32);
    } else {
      kernel = fmt::format("welford_inner_wide_{}_{}", in_str, out_str);
      tg_size = std::min(1024u, c10::metal::ceil_div(N, 32u) * 32u);
      if (N >= 2048) {
        tg_size = c10::metal::ceil_div(N / (4u * 16u), 32u) * 32u;
        tg_size = std::clamp(tg_size, 32u, 1024u);
      }
      num_tgs = M; // one TG per row
    }

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        auto ps = lib.getPipelineStateForFunc(kernel);
        getMPSProfiler().beginProfileKernel(ps, "welford_inner", {input});
        [enc setComputePipelineState:ps];
        struct {
          uint32_t M, N;
        } sizes_s = {M, N};
        mtl_setArgs(enc, input, output, mean_tensor, sizes_s, config);
        // 64-bit total dispatch thread count: num_tgs*tg_size can exceed 2^32
        // (one TG per row for many rows). num_tgs/tg_size stay 32-bit.
        [enc dispatchThreads:MTLSizeMake(static_cast<uint64_t>(num_tgs) * tg_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        getMPSProfiler().endProfileKernel(ps);
      }
    });
    return;
  }

  auto kernel = fmt::format("welford_{}_{}", in_str, out_str);
  auto params = build_reduce_params(input, reduce_dims, output, keepdim);

  uint32_t threads_per_group =
      std::min<uint32_t>(MAX_THREADGROUP_SIZE, c10::metal::ceil_div(reduction_size_u32, 32u) * 32u);
  // 64-bit total dispatch thread count: output.numel() * threads_per_group can
  // exceed 2^32 (one threadgroup per output element, many output elements).
  uint64_t num_threads = static_cast<uint64_t>(output.numel()) * threads_per_group;

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      auto ps = lib.getPipelineStateForFunc(kernel);
      getMPSProfiler().beginProfileKernel(ps, "welford_reduction", {input});
      [enc setComputePipelineState:ps];
      mtl_setArgs(enc, input, output, mean_tensor, params, config);
      [enc dispatchThreads:MTLSizeMake(num_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
      getMPSProfiler().endProfileKernel(ps);
    }
  });
}

static Tensor std_var_common_impl_mps(const Tensor& input_t,
                                      at::OptionalIntArrayRef dim,
                                      const std::optional<Scalar>& correction,
                                      bool keepdim,
                                      StdVarType stdVarType) {
  if (input_t.dim() == 0) {
    // Validate the user dim against the scalar before remapping to {0}; CPU
    // raises IndexError for e.g. dim=1 on a 0-d input, and dim_list_to_bitset
    // additionally raises on duplicate dims (e.g. dim=(0, 0)).
    if (dim.has_value()) {
      (void)at::dim_list_to_bitset(dim.value(), input_t.dim());
    }
    auto input_1d = input_t.unsqueeze(0);
    auto result = std_var_common_impl_mps(input_1d, IntArrayRef({0}), correction, false, stdVarType);
    return result.squeeze();
  }

  TORCH_CHECK(c10::isFloatingType(input_t.scalar_type()) || c10::isComplexType(input_t.scalar_type()),
              "std and var only support floating point and complex dtypes");
  if (c10::isComplexType(input_t.scalar_type())) {
    // Var(complex) = Var(real) + Var(imag) (same correction); std = sqrt.
    // Output is real-valued. Routes through the real welford kernels (no
    // complex Metal kernel), matching CPU/MPSGraph semantics.
    auto v = at::var(at::real(input_t).contiguous(), dim, correction, keepdim)
                 .add(at::var(at::imag(input_t).contiguous(), dim, correction, keepdim));
    return (stdVarType == STANDARD_DEVIATION) ? v.sqrt() : v;
  }

  auto reduce_dims = get_reduce_dims(input_t, dim);
  const auto correction_value = correction.value_or(1.0).toDouble();

  std::vector<int64_t> output_shape;
  for (int64_t d = 0; d < input_t.dim(); d++) {
    bool reduced = false;
    for (auto rd : reduce_dims) {
      if (rd == d) {
        reduced = true;
        break;
      }
    }
    if (reduced) {
      if (keepdim)
        output_shape.push_back(1);
    } else {
      output_shape.push_back(input_t.size(d));
    }
  }

  Tensor output_t = at::empty(output_shape, input_t.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);

  if (output_t.numel() == 0 || input_t.numel() == 0) {
    output_t.fill_(std::numeric_limits<float>::quiet_NaN());
    return output_t;
  }

  welford_kernel_mps(input_t, reduce_dims, keepdim, correction_value, stdVarType == STANDARD_DEVIATION, output_t);

  return output_t;
}

static Tensor min_max_mps_impl(const Tensor& input_t, MPSReductionType reduction_type, const std::string& func_name) {
  using CachedGraph = MPSUnaryCachedGraph;

  IntArrayRef input_shape = input_t.sizes();
  int64_t num_in_elements = c10::multiply_integers(input_shape);

  Tensor output_t = at::empty({}, input_t.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);

  if (output_t.numel() == 0 || num_in_elements == 0) {
    return output_t;
  }

  @autoreleasepool {
    std::string key = func_name + getTensorsStringKey(input_t);
    CachedGraph* cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_t);

      MPSGraphTensor* castOutputTensor = nil;
      MPSGraphTensor* castInputTensor = castToIHFTypes(mpsGraph, inputTensor, input_t);

      NSArray<NSNumber*>* axes = getTensorAxes(input_t);
      if (reduction_type == MPSReductionType::MAX) {
        castOutputTensor = [mpsGraph reductionMaximumPropagateNaNWithTensor:castInputTensor axes:axes name:nil];
      } else if (reduction_type == MPSReductionType::MIN) {
        castOutputTensor = [mpsGraph reductionMinimumPropagateNaNWithTensor:castInputTensor axes:axes name:nil];
      }

      MPSGraphTensor* outputTensor = castOutputTensor;
      if (getMPSDataType(output_t) != [castOutputTensor dataType]) {
        outputTensor = castMPSTensor(mpsGraph, castOutputTensor, output_t.scalar_type());
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t, @[ @1 ]);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return output_t;
}

static void min_max_out_mps(const Tensor& input_t,
                            int64_t dim,
                            bool keepdim,
                            const Tensor& output_t,
                            const Tensor& indices_t,
                            MPSReductionType reduction_type,
                            const std::string& func_name) {
  if (output_t.numel() == 0) {
    return;
  }
  if (input_t.numel() == 1 && input_t.dim() == 0) {
    output_t.fill_(input_t);
    indices_t.fill_(0);
    return;
  }

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
    MPSGraphTensor* indicesTensor_ = nil;
  };

  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());

  // Calculate the output shape according to keepdim=True
  // If there is no dim argument, the input shape is flattened
  IntArrayRef input_shape = input_t.sizes();
  int64_t num_input_dims = input_shape.size();
  NSMutableArray<NSNumber*>* apparent_out_shape = nil;

  apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
  for (const auto i : c10::irange(num_input_dims)) {
    apparent_out_shape[i] = dim_ == i ? @1 : [NSNumber numberWithInt:input_shape[i]];
  }

  auto stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = func_name + getTensorsStringKey({input_t, indices_t}) + ":" + std::to_string(dim_);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_t);
      MPSGraphTensor* outputTensor = nil;
      MPSGraphTensor* castInputTensor = castToIHFTypes(mpsGraph, inputTensor, input_t);

      if (reduction_type == MPSReductionType::MAX) {
        outputTensor = [mpsGraph reductionMaximumPropagateNaNWithTensor:castInputTensor axis:(NSInteger)dim_ name:nil];
      } else if (reduction_type == MPSReductionType::MIN) {
        outputTensor = [mpsGraph reductionMinimumPropagateNaNWithTensor:castInputTensor axis:(NSInteger)dim_ name:nil];
      }

      MPSGraphTensor* argreduceOutTensor = nil;
      if (reduction_type == MPSReductionType::MAX)
        argreduceOutTensor = [mpsGraph reductionArgMaximumWithTensor:castInputTensor
                                                                axis:(NSInteger)dim_
                                                                name:@"argmax_out"];
      else if (reduction_type == MPSReductionType::MIN)
        argreduceOutTensor = [mpsGraph reductionArgMinimumWithTensor:castInputTensor
                                                                axis:(NSInteger)dim_
                                                                name:@"argmax_out"];

      MPSGraphTensor* indicesTensor = nil;
      if ([argreduceOutTensor dataType] != MPSDataTypeInt64) {
        indicesTensor = [mpsGraph castTensor:argreduceOutTensor toType:MPSDataTypeInt64 name:@"cast_out"];
      }

      if ([outputTensor dataType] != getMPSDataType(output_t)) {
        outputTensor = castMPSTensor(mpsGraph, outputTensor, output_t.scalar_type());
      }
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
      newCachedGraph->indicesTensor_ = indicesTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t, apparent_out_shape);
    auto indicesPlaceholder = Placeholder(cachedGraph->indicesTensor_, indices_t, apparent_out_shape);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    auto results = dictionaryFromPlaceholders(outputPlaceholder, indicesPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}

// Min/Max with dim
static std::tuple<Tensor, Tensor> min_max_mps_impl(const Tensor& input_t,
                                                   int64_t dim,
                                                   bool keepdim,
                                                   MPSReductionType reduction_type,
                                                   const std::string& func_name) {
  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  native::zero_numel_check_dims(input_t, dim_, "max()");

  // Calculate the output shape according to keepdim=True
  // If there is no dim argument, the input shape is flattened
  IntArrayRef input_shape = input_t.sizes();
  int64_t num_input_dims = input_shape.size();
  NSMutableArray<NSNumber*>* apparent_out_shape = nil;
  // Use this if keepdim is false
  int64_t num_output_dims = num_input_dims - 1;

  std::vector<int64_t> vec_apparent_out_shape(num_input_dims);
  std::vector<int64_t> vec_out_shape(num_output_dims);

  apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
  // Counter for shape when keepdim is false
  int out_i = 0;
  for (const auto i : c10::irange(num_input_dims)) {
    if (dim_ == i) {
      apparent_out_shape[i] = @1;
      vec_apparent_out_shape[i] = 1;
    } else {
      apparent_out_shape[i] = [NSNumber numberWithInt:input_shape[i]];
      vec_apparent_out_shape[i] = input_shape[i];
      vec_out_shape[out_i] = input_shape[i];
      out_i++;
    }
  }

  Tensor output_t;
  Tensor indices_t;
  if (!keepdim) {
    output_t =
        at::empty(IntArrayRef(vec_out_shape), input_t.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
    indices_t = at::empty(IntArrayRef(vec_out_shape), ScalarType::Long, std::nullopt, kMPS, std::nullopt, std::nullopt);
  } else {
    output_t = at::empty(
        IntArrayRef(vec_apparent_out_shape), input_t.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
    indices_t = at::empty(
        IntArrayRef(vec_apparent_out_shape), ScalarType::Long, std::nullopt, kMPS, std::nullopt, std::nullopt);
  }

  if (output_t.numel() == 0 || input_t.numel() == 0) {
    return std::tuple<Tensor, Tensor>{output_t, indices_t};
  }

  min_max_out_mps(input_t, dim, keepdim, output_t, indices_t, reduction_type, func_name);

  return std::tuple<Tensor, Tensor>{output_t, indices_t};
}

static void argmax_argmin_out_mps(const Tensor& input_t,
                                  std::optional<int64_t> dim,
                                  bool keepdim,
                                  const Tensor& output_t,
                                  MPSReductionType reduction_type,
                                  const std::string& func_name) {
  const bool is_argmax = (reduction_type == MPSReductionType::MAX);
  const char* op_name = is_argmax ? "argmax()" : "argmin()";

  int64_t dim_ = -1;
  if (dim.has_value()) {
    dim_ = maybe_wrap_dim(dim.value(), input_t.dim());
    zero_numel_check_dims(input_t, dim_, op_name);
  } else {
    TORCH_CHECK_INDEX(
        input_t.numel() != 0, op_name, ": Expected reduction dim to be specified for input.numel() == 0.");
  }

  if (output_t.numel() == 0) {
    return;
  }
  // 0-dim input: only index 0 is reachable.
  if (input_t.dim() == 0) {
    output_t.fill_(0);
    return;
  }

  // For full reduction (dim==None) we materialize a contiguous 1-D view so the
  // returned linear index follows the standard "as-if-contiguous" convention,
  // regardless of input strides.
  Tensor input;
  Tensor output_view;
  int64_t reduce_dim = 0;
  if (dim.has_value()) {
    input = input_t;
    output_view = keepdim ? output_t : output_t.unsqueeze(dim_);
    reduce_dim = dim_;
  } else {
    input = input_t.contiguous().view(-1);
    output_view = output_t.view({1});
  }
  TORCH_CHECK(static_cast<uint32_t>(input.dim()) <= c10::metal::max_ndim,
              func_name,
              ": tensor rank > ",
              c10::metal::max_ndim,
              " is not supported");

  // Metal has no simd_min/max for bool; remap to 1-byte char (identical 0/1
  // layout). Complex types have no ordering, so argmax/argmin is undefined.
  ScalarType in_kdtype = input.scalar_type();
  TORCH_CHECK(!c10::isComplexType(in_kdtype), func_name, ": not implemented for ", in_kdtype);
  if (in_kdtype == kBool) {
    in_kdtype = kChar;
  }
  const auto op_prefix = is_argmax ? "argmax" : "argmin";
  const auto in_str = scalarToMetalTypeString(in_kdtype);
  MPSStream* stream = getCurrentMPSStream();

  // Fast paths: when the reduced dim is the outermost or innermost dim of a
  // contiguous input (and the output is contiguous), dispatch a specialized
  // kernel with a tuned grid layout, mirroring value_reduction_outer /
  // value_reduction_inner.
  if (dim.has_value() && input.is_contiguous() && output_t.is_contiguous() && input.dim() >= 2 &&
      (reduce_dim == 0 || reduce_dim == input.dim() - 1)) {
    const bool is_outer = (reduce_dim == 0);
    const uint32_t M = is_outer ? static_cast<uint32_t>(input.size(0))
                                : static_cast<uint32_t>(input.numel() / input.size(input.dim() - 1));
    const uint32_t N = is_outer ? static_cast<uint32_t>(input.numel() / input.size(0))
                                : static_cast<uint32_t>(input.size(input.dim() - 1));
    const auto kernel_name = fmt::format("{}_reduction_{}_{}_long", op_prefix, is_outer ? "outer" : "inner", in_str);
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> ce = stream->commandEncoder();
        auto ps = lib.getPipelineStateForFunc(kernel_name);
        getMPSProfiler().beginProfileKernel(ps, func_name, {input});
        [ce setComputePipelineState:ps];
        if (is_outer) {
          constexpr uint32_t TG_X = 32, TG_Y = 32;
          // 4th element is trailing pad so the host-side bind matches the
          // kernel's `constant uint3&` slot (uint3 has 16-byte alignment in
          // Metal even though only 12 bytes are read). Without this Metal
          // API validation flags a buffer-length mismatch.
          const std::array<uint32_t, 4> sizes_s{M, N, 1, 0};
          mtl_setArgs(ce, input, output_t, sizes_s);
          const auto num_tg_x = c10::metal::ceil_div(N, TG_X);
          [ce dispatchThreads:MTLSizeMake(num_tg_x * TG_X, TG_Y, 1) threadsPerThreadgroup:MTLSizeMake(TG_X, TG_Y, 1)];
        } else {
          constexpr uint32_t TG_SIZE = 256;
          constexpr uint32_t rows_per_tg = TG_SIZE / 32;
          const auto num_tgs = c10::metal::ceil_div(M, rows_per_tg);
          struct {
            uint32_t M, N;
          } sizes_s = {M, N};
          mtl_setArgs(ce, input, output_t, sizes_s);
          [ce dispatchThreads:MTLSizeMake(num_tgs * TG_SIZE, 1, 1) threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
        }
        getMPSProfiler().endProfileKernel(ps);
      }
    });
    return;
  }

  const auto kernel_name = fmt::format("{}_reduction_{}_long", op_prefix, in_str);

  NormParams params{};
  params.ndim = input.dim();
  params.reduction_size = static_cast<uint32_t>(input.size(reduce_dim));
  for (const auto d : c10::irange(input.dim())) {
    params.input_sizes[d] = input.size(d);
    params.input_strides[d] = input.stride(d);
    params.output_sizes[d] = output_view.size(d);
    params.output_strides[d] = output_view.stride(d);
  }

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> ce = stream->commandEncoder();
      auto ps = lib.getPipelineStateForFunc(kernel_name);
      getMPSProfiler().beginProfileKernel(ps, func_name, {input});
      [ce setComputePipelineState:ps];
      mtl_setArgs(ce, input, output_view, params);
      // Pad per-TG thread count up to a full simdgroup; padding lanes load
      // Op::identity() and skip the per-thread scan, keeping the two-stage
      // SIMD reduction well-defined for all reduction sizes.
      const auto threads_per_group = std::min(MAX_THREADGROUP_SIZE, c10::metal::round_up(params.reduction_size, 32u));
      const auto num_threads = static_cast<uint32_t>(output_view.numel()) * threads_per_group;
      [ce dispatchThreads:MTLSizeMake(num_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
      getMPSProfiler().endProfileKernel(ps);
    }
  });
}

// Unified host-side dispatch for value-preserving reductions on MPS, shared
// by sum/nansum/mean/count_nonzero and min/max/all/any. Kernel name pattern
// is always `{prefix}reduction_{variant}_{TI}_{TO}` with variant in
// `""/"outer"/"inner"`. Selects among four code paths:
//   1. Outer-dim kernel (dim=0 on contiguous input).
//   2. Inner-dim kernel (last dim on contiguous input).
//   3. Two-pass full reduction (scalar output, large input).
//   4. Generic single-pass fallback.
struct ReductionDispatch {
  std::string prefix; // "sum_", "nansum_", "count_nonzero_", "min_", "max_",
                      // "all_", "any_".
  ScalarType input_kernel_dtype; // may differ from input.scalar_type() (e.g.
                                 // bool -> char for min/max).
  ScalarType output_kernel_dtype; // may differ from output.scalar_type() for
                                  // the same remap reason.
  ScalarType partial_dtype; // pass-1 output dtype: output.scalar_type() for
                            // sum/min/max, uchar for all/any.
  std::string pass2_prefix; // pass-2 op prefix. count_nonzero -> "sum_" (the
                            // partials are already per-block counts), all/any
                            // -> "min_"/"max_" (predicate ran in pass 1).
  bool has_strided_pass1 = false; // sum has a `_strided_` pass-1 kernel; ops
                                  // without it call .contiguous() first.
  std::optional<float> divisor; // sum/mean only; appended as a float buffer
                                // to the outer/inner kernel signatures, and
                                // passed via NormParams.p elsewhere.
  bool inner_specializations = false; // op registers inner_thin/inner_wide
                                      // kernels for degenerate inner shapes
                                      // (prod); others use the generic inner.
};

static void reduction_dispatch_mps(TensorIterator& iter, const ReductionDispatch& opts) {
  Tensor output = iter.output(0);
  Tensor input_orig = iter.input(0);
  TORCH_INTERNAL_ASSERT(input_orig.numel() > 0 && output.numel() > 0);
  TORCH_INTERNAL_ASSERT(output.dim() == input_orig.dim());
  if (!input_orig.is_contiguous()) {
    c10::DimVector perm(input_orig.dim());
    std::iota(perm.begin(), perm.end(), 0);
    std::ranges::stable_sort(perm, std::greater{}, [&](int64_t d) { return input_orig.stride(d); });
    auto permuted = input_orig.permute(perm);
    if (permuted.is_contiguous()) {
      input_orig = std::move(permuted);
      output = output.permute(perm);
    }
  }

  const uint32_t reduction_size = input_orig.numel() / output.numel();
  constexpr uint32_t NCHAINS = SUM_NCHAINS;
  MPSStream* stream = getCurrentMPSStream();

  const auto in_str = scalarToMetalTypeString(opts.input_kernel_dtype);
  const auto out_str = scalarToMetalTypeString(opts.output_kernel_dtype);
  const auto partial_str = scalarToMetalTypeString(opts.partial_dtype);
  // Two-pass reductions accumulate per-chunk partials in opmath (fp32 for
  // half/bfloat) so a chunk product can exceed the narrow output range without
  // overflowing -- the full product is finite. Pass 2 narrows opmath -> output.
  const ScalarType opmath_pdt =
      (opts.partial_dtype == kHalf || opts.partial_dtype == kBFloat16) ? kFloat : opts.partial_dtype;
  const auto opmath_str = scalarToMetalTypeString(opmath_pdt);
  // complexHalf opmath (float2) partials are not wired for the two-pass; those
  // keep the fp32-accumulating single-pass kernels.
  const auto out_dt = output.scalar_type();
  const bool two_pass_ok = out_dt != kComplexHalf;

  // Outer-dim (dim=0 on contiguous input) and inner-dim (last dim on
  // contiguous input) specializations: handle the dim-reduction case with
  // dedicated kernels that have better thread layout than the generic kernel.
  if (output.numel() > 1 && input_orig.is_contiguous() && output.is_contiguous()) {
    int num_reduced = 0;
    int reduced_dim = -1;
    for (int64_t d = 0; d < input_orig.dim(); d++) {
      if (input_orig.size(d) != output.size(d)) {
        num_reduced++;
        reduced_dim = d;
      }
    }
    if (num_reduced == 1 && reduced_dim == 0 && input_orig.dim() >= 2) {
      uint32_t M = input_orig.size(0);
      uint32_t N = input_orig.numel() / M;
      // Outer bucketed two-pass: few output columns (small N) over a long
      // reduced dim (large M). The single-pass outer kernel launches
      // ceil(N/32) threadgroups -- one for N<=32 -- with TG_Y=32 lanes each
      // serially walking M/32 rows, leaving the GPU idle. Pass 1 instead splits
      // the flat M*N input into contiguous N-aligned slices, folds each element
      // into its column bucket, and writes num_tgs*N opmath partials; pass 2
      // reduces those partials per column. Only worth it for very few output
      // columns over a very long reduced dim; the two-pass overhead loses on
      // moderate shapes the single-pass kernel already parallelizes adequately.
      if (opts.inner_specializations && two_pass_ok && N <= 8 && M >= (1u << 18) && !opts.divisor.has_value()) {
        // The bucketed kernel folds elements into N column buckets in a fixed
        // acc[MAXN=8] register array; routing N > 8 here would index out of
        // bounds, so the gate's N <= 8 is a hard kernel-domain invariant.
        TORCH_INTERNAL_ASSERT(N <= 8, "bucketed prod kernel handles at most 8 columns");
        // pass 1: bucketed coalesced read of the flat M*N input -> num_tgs*N
        // opmath partials; pass 2: the single-pass outer kernel collapses the
        // num_tgs partials per column (num_tgs is small, so it is fast there).
        const uint64_t total = (uint64_t)M * N;
        constexpr uint32_t TG = 256;
        // Fewer, larger threadgroups: each does one N-bucket reduce over a big
        // contiguous slice, so the reduce cost is amortized over more reads (the
        // reduce, not bandwidth, is the floor here). Still enough TGs to fill the
        // GPU.
        const uint32_t num_tgs = std::clamp<uint32_t>(static_cast<uint32_t>(total / 32768u), 64u, 1024u);
        auto partials = at::empty({(int64_t)num_tgs * N}, output.options().dtype(opmath_pdt));
        auto p1 = fmt::format("{}reduction_outer_bucketed_{}_{}", opts.prefix, in_str, opmath_str);
        auto p2 = fmt::format("{}reduction_outer_{}_{}", opts.pass2_prefix, opmath_str, out_str);
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          @autoreleasepool {
            id<MTLComputeCommandEncoder> ce = stream->commandEncoder();
            auto ps1 = lib.getPipelineStateForFunc(p1);
            getMPSProfiler().beginProfileKernel(ps1, opts.prefix + "reduction_outer_bucketed", {input_orig});
            // 4th element pads the bind to 16 bytes: Metal's uint3 is 16-byte
            // aligned even though the kernel reads only 12 (as the outer path does).
            const std::array<uint32_t, 4> sizes1{static_cast<uint32_t>(total), N, num_tgs, 0};
            [ce setComputePipelineState:ps1];
            mtl_setArgs(ce, input_orig, partials, sizes1);
            [ce dispatchThreads:MTLSizeMake((int64_t)num_tgs * TG, 1, 1) threadsPerThreadgroup:MTLSizeMake(TG, 1, 1)];
            getMPSProfiler().endProfileKernel(ps1);

            // pass 2: (num_tgs, N) reduce dim=0 -> output[N] via the outer kernel
            auto ps2 = lib.getPipelineStateForFunc(p2);
            getMPSProfiler().beginProfileKernel(ps2, opts.prefix + "reduction_outer_pass2", {partials});
            constexpr uint32_t TG_X = 32, TG_Y = 32;
            const std::array<uint32_t, 4> sizes2{num_tgs, N, 1, 0};
            [ce setComputePipelineState:ps2];
            mtl_setArgs(ce, partials, output, sizes2);
            const auto num_tg_x = c10::metal::ceil_div(N, TG_X);
            [ce dispatchThreads:MTLSizeMake(num_tg_x * TG_X, TG_Y, 1) threadsPerThreadgroup:MTLSizeMake(TG_X, TG_Y, 1)];
            getMPSProfiler().endProfileKernel(ps2);
          }
        });
        return;
      }
      // Thin outer: a short reduced dim (small M) over many output columns
      // (large N). The TG-tiled outer kernel idles TG_Y-M of its 32 row-workers
      // and still launches ceil(N/32) threadgroups; one thread per column reads
      // coalesced across adjacent gid and fully occupies the GPU.
      if (opts.inner_specializations && M <= 32 && N >= 1024 && !opts.divisor.has_value()) {
        auto thin_kernel = fmt::format("{}reduction_outer_thin_{}_{}", opts.prefix, in_str, out_str);
        constexpr uint32_t TG = 256;
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          @autoreleasepool {
            id<MTLComputeCommandEncoder> ce = stream->commandEncoder();
            auto ps = lib.getPipelineStateForFunc(thin_kernel);
            getMPSProfiler().beginProfileKernel(ps, opts.prefix + "reduction_outer_thin", {input_orig});
            struct {
              uint32_t M, N;
            } sizes_s = {M, N};
            [ce setComputePipelineState:ps];
            mtl_setArgs(ce, input_orig, output, sizes_s);
            [ce dispatchThreads:MTLSizeMake(c10::metal::ceil_div(N, TG) * TG, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(TG, 1, 1)];
            getMPSProfiler().endProfileKernel(ps);
          }
        });
        return;
      }
      auto outer_kernel = fmt::format("{}reduction_outer_{}_{}", opts.prefix, in_str, out_str);
      constexpr uint32_t TG_X = 32, TG_Y = 32;
      const auto num_tg_x = c10::metal::ceil_div(N, TG_X);
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          id<MTLComputeCommandEncoder> ce = stream->commandEncoder();
          auto ps = lib.getPipelineStateForFunc(outer_kernel);
          getMPSProfiler().beginProfileKernel(ps, opts.prefix + "reduction_outer", {input_orig});
          // 4th element is trailing pad so the host-side bind matches the
          // kernel's `constant uint3&` slot (16-byte alignment in Metal even
          // though only 12 bytes are read).
          const std::array<uint32_t, 4> sizes_s{M, N, 1, 0};
          [ce setComputePipelineState:ps];
          if (opts.divisor.has_value()) {
            mtl_setArgs(ce, input_orig, output, sizes_s, *opts.divisor);
          } else {
            mtl_setArgs(ce, input_orig, output, sizes_s);
          }
          [ce dispatchThreads:MTLSizeMake(num_tg_x * TG_X, TG_Y, 1) threadsPerThreadgroup:MTLSizeMake(TG_X, TG_Y, 1)];
          getMPSProfiler().endProfileKernel(ps);
        }
      });
      return;
    }
    if (num_reduced == 1 && reduced_dim == input_orig.dim() - 1) {
      uint32_t N = input_orig.size(input_orig.dim() - 1);
      uint32_t M = input_orig.numel() / N;
      // Super-wide: few rows (small M) with a huge reduced dim. One TG per row
      // (inner_wide) leaves most of the GPU idle and is bandwidth-starved. Split
      // each row into K chunks -> M*K parallel partials (pass 1), then reduce the
      // K partials per row (pass 2). Reuses the generic NormParams value_reduction
      // kernel, mirroring the full-reduce two-pass but keeping the M batch dim.
      // Partials are stored at opmath (fp32 for half/bfloat, via opmath_pdt) so
      // a per-chunk product stays lossless even when it would overflow the
      // narrow output range (the full product is finite). complexHalf opmath
      // (float2) is not wired for the two-pass, so it keeps inner_wide (one
      // TG/row, fp32 accumulator).
      if (opts.inner_specializations && two_pass_ok && M < 32 && N >= 4096 && !opts.divisor.has_value()) {
        uint32_t K = std::clamp<uint32_t>(512u / std::max<uint32_t>(M, 1u), 1u, 512u);
        while (K > 1 && N % K != 0) {
          K--;
        }
        if (K > 1) {
          const uint32_t chunkN = N / K;
          auto input_c = input_orig.contiguous();
          auto partials = at::empty({(int64_t)M * K}, output.options().dtype(opmath_pdt));
          auto p1 = fmt::format("{}reduction_{}_{}", opts.prefix, in_str, opmath_str);
          auto p2 = fmt::format("{}reduction_{}_{}", opts.pass2_prefix, opmath_str, out_str);
          // pass 1: [M*K, chunkN] reduce dim=1 -> partials[M*K]
          NormParams params1{};
          params1.ndim = 2;
          params1.reduction_size = chunkN;
          params1.input_sizes[0] = M * K;
          params1.input_strides[0] = chunkN;
          params1.input_sizes[1] = chunkN;
          params1.input_strides[1] = 1;
          params1.output_sizes[0] = M * K;
          params1.output_strides[0] = 1;
          // pass 2: partials[M, K] reduce dim=1 -> output[M]
          NormParams params2{};
          params2.ndim = 2;
          params2.reduction_size = K;
          params2.input_sizes[0] = M;
          params2.input_strides[0] = K;
          params2.input_sizes[1] = K;
          params2.input_strides[1] = 1;
          params2.output_sizes[0] = M;
          params2.output_strides[0] = 1;
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            @autoreleasepool {
              id<MTLComputeCommandEncoder> ce = stream->commandEncoder();
              auto ps1 = lib.getPipelineStateForFunc(p1);
              getMPSProfiler().beginProfileKernel(ps1, opts.prefix + "reduction_wide_pass1", {input_c});
              [ce setComputePipelineState:ps1];
              mtl_setArgs(ce, input_c, partials, params1);
              auto tpg1 = std::min<uint32_t>(MAX_THREADGROUP_SIZE, c10::metal::round_up(chunkN, 32u));
              [ce dispatchThreads:MTLSizeMake((int64_t)M * K * tpg1, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tpg1, 1, 1)];
              getMPSProfiler().endProfileKernel(ps1);

              auto ps2 = lib.getPipelineStateForFunc(p2);
              getMPSProfiler().beginProfileKernel(ps2, opts.prefix + "reduction_wide_pass2", {partials});
              [ce setComputePipelineState:ps2];
              mtl_setArgs(ce, partials, output, params2);
              auto tpg2 = std::min<uint32_t>(MAX_THREADGROUP_SIZE, c10::metal::round_up(K, 32u));
              [ce dispatchThreads:MTLSizeMake((int64_t)M * tpg2, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg2, 1, 1)];
              getMPSProfiler().endProfileKernel(ps2);
            }
          });
          return;
        }
      }
      // Shape-specialized inner kernels, only for ops that register them
      // (inner_specializations): thin = one thread per row for a tiny reduced
      // dim; wide = one whole TG per row for few huge rows; else the generic
      // simd-per-row kernel. Non-specialized ops keep the generic path.
      std::string inner_kernel;
      uint32_t tg_size;
      int64_t num_tgs;
      if (opts.inner_specializations && M >= 64 && N <= 128) {
        inner_kernel = fmt::format("{}reduction_inner_thin_{}_{}", opts.prefix, in_str, out_str);
        tg_size = 256;
        num_tgs = c10::metal::ceil_div<int64_t>(M, tg_size); // one thread per row
      } else if (!opts.inner_specializations || M >= 64) {
        inner_kernel = fmt::format("{}reduction_inner_{}_{}", opts.prefix, in_str, out_str);
        // One simdgroup (32 lanes) per row, 8 rows per 256-thread TG. A larger TG
        // would pack more rows per TG and shrink num_tgs (ceil(M/(tg/32))),
        // under-utilizing the GPU for moderate M, so keep 256 for all ops.
        tg_size = 256u;
        num_tgs = c10::metal::ceil_div<int64_t>(M, tg_size / 32);
      } else {
        inner_kernel = fmt::format("{}reduction_inner_wide_{}_{}", opts.prefix, in_str, out_str);
        tg_size = std::min<uint32_t>(1024u, c10::metal::ceil_div(N, 32u) * 32u);
        if (N >= 2048) {
          tg_size = c10::metal::ceil_div(N / (4u * 16u), 32u) * 32u;
          tg_size = std::clamp<uint32_t>(tg_size, 32u, 1024u);
        }
        num_tgs = M; // one TG per row
      }
      const int64_t total_threads = num_tgs * tg_size;
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          id<MTLComputeCommandEncoder> ce = stream->commandEncoder();
          auto ps = lib.getPipelineStateForFunc(inner_kernel);
          getMPSProfiler().beginProfileKernel(ps, opts.prefix + "reduction_inner", {input_orig});
          struct {
            uint32_t M, N;
          } sizes_s = {M, N};
          [ce setComputePipelineState:ps];
          if (opts.divisor.has_value()) {
            mtl_setArgs(ce, input_orig, output, sizes_s, *opts.divisor);
          } else {
            mtl_setArgs(ce, input_orig, output, sizes_s);
          }
          [ce dispatchThreads:MTLSizeMake(total_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
          getMPSProfiler().endProfileKernel(ps);
        }
      });
      return;
    }
  }

  // Two-pass for large full reductions: pass 1 splits input into <=512
  // contiguous slices, each TG reduces one slice to a partial; pass 2 collapses
  // the num_groups partials into the final scalar. prod's complexHalf is excluded
  // (its float2 opmath partials are not wired here, and half2 partials would
  // overflow on a long slice); it falls to the fp32-accumulating single-pass.
  if (output.numel() == 1 && reduction_size > MAX_THREADGROUP_SIZE * NCHAINS &&
      !(opts.inner_specializations && output.scalar_type() == kComplexHalf)) {
    // Keep pass-2 overhead proportional to input size. Small full reductions are
    // latency-bound and prefer fewer partials; large ones need enough pass-1
    // parallelism to stay bandwidth-bound.
    uint32_t max_num_groups = 384u;
    if (reduction_size <= (1u << 20)) {
      max_num_groups = 32u;
    } else if (reduction_size <= (1u << 22)) {
      max_num_groups = 192u;
    }
    auto num_groups = std::min(max_num_groups, c10::metal::ceil_div(reduction_size, MAX_THREADGROUP_SIZE * NCHAINS));
    while (num_groups > 1 && reduction_size % num_groups != 0) {
      num_groups--;
    }
    if (num_groups > 1) {
      const bool is_contig = input_orig.is_contiguous();
      // For ops without a strided pass-1 kernel, .contiguous() the input
      // (no-op when already contiguous).
      auto input = (!is_contig && !opts.has_strided_pass1) ? input_orig.contiguous() : input_orig;
      const bool use_strided = !is_contig && opts.has_strided_pass1;
      const uint32_t elems_per_group = reduction_size / num_groups;
      // prod stores opmath (fp32 for half/bfloat) pass-1 partials so a per-slice
      // product can exceed the narrow output range without overflowing -- the
      // full product is finite. Other ops keep output-dtype partials; gated on
      // inner_specializations because only prod registers the opmath fp32->out
      // pass-2 kernel.
      const auto sc_pdt = opts.inner_specializations ? opmath_pdt : opts.partial_dtype;
      const auto sc_str = opts.inner_specializations ? opmath_str : partial_str;
      auto partials = at::empty({(int64_t)num_groups}, output.options().dtype(sc_pdt));

      auto p1_kernel = fmt::format("{}reduction{}_{}_{}", opts.prefix, use_strided ? "_strided" : "", in_str, sc_str);
      auto p2_kernel = fmt::format("{}reduction_{}_{}", opts.pass2_prefix, sc_str, out_str);

      NormParams params1{};
      params1.reduction_size = elems_per_group;
      if (use_strided) {
        params1.ndim = input.dim();
        for (const auto d : c10::irange(input.dim())) {
          params1.input_sizes[d] = input.size(d);
          params1.input_strides[d] = input.stride(d);
        }
      } else {
        // Model as 2D: input is [num_groups, elems_per_group], reduce dim=1.
        params1.ndim = 2;
        params1.input_sizes[0] = num_groups;
        params1.input_strides[0] = elems_per_group;
        params1.output_sizes[0] = num_groups;
        params1.output_strides[0] = 1;
        params1.input_sizes[1] = elems_per_group;
        params1.input_strides[1] = 1;
      }

      // Pass 2: partials[num_groups] -> output[1], reduce dim=0. divisor
      // applies here (not on pass 1) so the accumulator/divisor happens in
      // opmath_t before the final cast to output dtype.
      NormParams params2{};
      params2.ndim = 1;
      params2.p = opts.divisor.value_or(0.0f);
      params2.reduction_size = num_groups;
      params2.input_sizes[0] = num_groups;
      params2.input_strides[0] = 1;
      params2.output_sizes[0] = 1;
      params2.output_strides[0] = 0;

      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          id<MTLComputeCommandEncoder> ce = stream->commandEncoder();

          auto ps1 = lib.getPipelineStateForFunc(p1_kernel);
          getMPSProfiler().beginProfileKernel(ps1, opts.prefix + "reduction_pass1", {input});
          [ce setComputePipelineState:ps1];
          mtl_setArgs(ce, input, partials, params1);
          // Round both passes' TG sizes up to a full simdgroup. Required
          // because c10::metal::simd_max/min<long> emulates 64-bit simd
          // via simd_shuffle_and_fill_down, and inactive lanes (when active
          // count < 32) return undefined data (in practice 0) instead of
          // the op's identity — corrupting min/max of all-positive or
          // all-negative longs. The fill value itself is fixed in
          // reduction_utils.h, but the fill only applies to past-end
          // shuffles, not to inactive-lane reads within the simdgroup.
          // Padding threads here skip the load loop (tid >= rsize) and
          // contribute Op::identity().
          auto tpg1 = std::min(MAX_THREADGROUP_SIZE, c10::metal::round_up(elems_per_group, 32u));
          [ce dispatchThreads:MTLSizeMake(num_groups * tpg1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg1, 1, 1)];
          getMPSProfiler().endProfileKernel(ps1);

          auto ps2 = lib.getPipelineStateForFunc(p2_kernel);
          getMPSProfiler().beginProfileKernel(ps2, opts.prefix + "reduction_pass2", {partials});
          [ce setComputePipelineState:ps2];
          mtl_setArgs(ce, partials, output, params2);
          auto tpg2 = std::min(MAX_THREADGROUP_SIZE, c10::metal::round_up(num_groups, 32u));
          [ce dispatchThreads:MTLSizeMake(tpg2, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg2, 1, 1)];
          getMPSProfiler().endProfileKernel(ps2);
        }
      });
      return;
    }
  }

  // Generic single-pass fallback.
  auto kernel_name = fmt::format("{}reduction_{}_{}", opts.prefix, in_str, out_str);
  NormParams params{};
  params.ndim = input_orig.dim();
  params.p = opts.divisor.value_or(0.0f);
  params.reduction_size = reduction_size;
  for (const auto dim_idx : c10::irange(input_orig.dim())) {
    params.input_sizes[dim_idx] = input_orig.size(dim_idx);
    params.input_strides[dim_idx] = input_orig.stride(dim_idx);
    params.output_sizes[dim_idx] = output.size(dim_idx);
    params.output_strides[dim_idx] = output.stride(dim_idx);
  }
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> ce = stream->commandEncoder();
      auto ps = lib.getPipelineStateForFunc(kernel_name);
      getMPSProfiler().beginProfileKernel(ps, opts.prefix + "reduction", {input_orig});
      [ce setComputePipelineState:ps];
      mtl_setArgs(ce, input_orig, output, params);
      // Round per-TG thread count up to a full simdgroup (32 lanes). With
      // fewer threads, inactive lanes still participate in simd_shuffle but
      // carry register-zero, corrupting min/max reductions whose identity
      // is not zero. Padding threads load Op::identity() and contribute
      // nothing to the result.
      const auto threads_per_group = std::min(MAX_THREADGROUP_SIZE, c10::metal::round_up(reduction_size, 32u));
      uint32_t num_threads = output.numel() * threads_per_group;
      [ce dispatchThreads:MTLSizeMake(num_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
      getMPSProfiler().endProfileKernel(ps);
    }
  });
}

// Shared implementation for sum/nansum/count_nonzero/mean. `divisor` > 0
// divides the accumulator (in opmath_t) before casting to output, enabling
// fused mean.
static void sum_nansum_kernel_mps(TensorIterator& iter, const std::string& kernel_prefix, float divisor = 0.0f) {
  const Tensor& input = iter.input(0);
  const Tensor& output = iter.output(0);
  if (input.numel() == 0) {
    output.zero_();
    return;
  }
  if (output.numel() == 0) {
    return;
  }
  // Pass 2 always sums partials (count_nonzero's partials are per-block
  // counts -- counting again would be wrong, so always use sum_).
  reduction_dispatch_mps(iter,
                         ReductionDispatch{
                             .prefix = kernel_prefix,
                             .input_kernel_dtype = input.scalar_type(),
                             .output_kernel_dtype = output.scalar_type(),
                             .partial_dtype = output.scalar_type(),
                             .pass2_prefix = "sum_",
                             .has_strided_pass1 = true,
                             .divisor = divisor,
                         });
}

static void sum_kernel_mps(TensorIterator& iter) {
  sum_nansum_kernel_mps(iter, "sum_");
}

static void nansum_kernel_mps(TensorIterator& iter) {
  auto in_dtype = iter.input(0).scalar_type();
  bool is_float = c10::isFloatingType(in_dtype) || c10::isComplexType(in_dtype);
  sum_nansum_kernel_mps(iter, is_float ? "nansum_" : "sum_");
}

static void mean_kernel_mps(TensorIterator& iter) {
  auto output = iter.output(0);
  auto input = iter.input(0);
  if (input.numel() == 0 || output.numel() == 0) {
    sum_nansum_kernel_mps(iter, "sum_");
    return;
  }
  int64_t reduction_size = input.numel() / output.numel();
  // Fused divide: the sum kernel divides the accumulator (in opmath_t)
  // before casting to output, so fp32 accumulation precision is preserved
  // for fp16/bf16/half2 without an intermediate tensor.
  sum_nansum_kernel_mps(iter, "sum_", static_cast<float>(reduction_size));
}

static void count_nonzero_kernel_mps(TensorIterator& iter) {
  sum_nansum_kernel_mps(iter, "count_nonzero_");
}

// Value reductions: min/max (Op + identity load on T), all/any (Op +
// predicate load with uchar accumulator). Delegates to the shared
// reduction_dispatch_mps.
static void value_reduction_kernel_mps(TensorIterator& iter, const std::string& op_prefix) {
  const Tensor& input = iter.input(0);
  const Tensor& output = iter.output(0);
  if (input.numel() == 0 || output.numel() == 0) {
    return;
  }
  const bool is_predicate = op_prefix == "all_" || op_prefix == "any_";
  // For min/max, Metal's simd_min/simd_max have no bool overload; remap
  // BOTH input and output to char (identical 1-byte 0/1 layout). all/any
  // outputs uchar partials regardless of input dtype.
  ScalarType in_kdtype = input.scalar_type();
  ScalarType out_kdtype = output.scalar_type();
  if (!is_predicate && in_kdtype == kBool) {
    in_kdtype = out_kdtype = kChar;
  } else if (is_predicate) {
    out_kdtype = kByte;
  }
  // all/any partials are uchar (the predicate-reduction accumulator); pass 2
  // collapses uchar partials with min/max. For min/max, partial == output.
  ScalarType partial_dtype = is_predicate ? kByte : out_kdtype;
  std::string pass2_prefix = op_prefix;
  if (op_prefix == "all_") {
    pass2_prefix = "min_";
  } else if (op_prefix == "any_") {
    pass2_prefix = "max_";
  }
  reduction_dispatch_mps(iter,
                         ReductionDispatch{
                             .prefix = op_prefix,
                             .input_kernel_dtype = in_kdtype,
                             .output_kernel_dtype = out_kdtype,
                             .partial_dtype = partial_dtype,
                             .pass2_prefix = pass2_prefix,
                         });
}

static void min_values_kernel_mps(TensorIterator& iter) {
  value_reduction_kernel_mps(iter, "min_");
}

static void max_values_kernel_mps(TensorIterator& iter) {
  value_reduction_kernel_mps(iter, "max_");
}

static void and_kernel_mps(TensorIterator& iter) {
  value_reduction_kernel_mps(iter, "all_");
}

static void or_kernel_mps(TensorIterator& iter) {
  value_reduction_kernel_mps(iter, "any_");
}

Tensor trace_mps(const Tensor& self) {
  TORCH_CHECK(self.dim() == 2, "trace: expected a matrix, but got tensor with dim ", self.dim());
  // trace is just sum-of-diagonal; route through the Metal sum kernel via
  // .diagonal().sum() instead of a dedicated MPSGraph reduction.
  return self.diagonal().sum();
}

// (input, output) dtype pairs with a registered prod value_reduction kernel.
static bool prod_kernel_registered(ScalarType in, ScalarType out) {
  if (in == out) {
    return in == kFloat || in == kHalf || in == kBFloat16 || in == kInt || in == kLong || in == kShort || in == kChar ||
        in == kByte || in == kComplexFloat || in == kComplexHalf;
  }
  if ((in == kHalf || in == kBFloat16) && out == kFloat) {
    return true;
  }
  if ((in == kInt || in == kShort || in == kChar || in == kByte) && out == kLong) {
    return true;
  }
  if (in == kBool && (out == kInt || out == kLong)) {
    return true;
  }
  return false;
}

// prod via the shared value_reduction<ProdOp> kernel: route the dtype semantics
// (dtype= cast, bool nonzero-product, uint rejection), then dispatch through the
// same reduction_dispatch_mps sum/min/max use. pass-2 multiplies the partials.
static void prod_kernel_mps(const Tensor& input, IntArrayRef dims, bool keepdim, const Tensor& output) {
  // bool output: prod(..., dtype=bool) is nonzero-product (CPU multiplies the
  // input != 0 mask). Reduce input.to(kBool) via the (bool,long) kernel, cast
  // the long accumulator back to bool.
  if (output.scalar_type() == kBool) {
    auto long_output = at::empty_like(output, output.options().dtype(kLong));
    prod_kernel_mps(input.to(kBool), dims, keepdim, long_output);
    output.copy_(long_output.to(kBool));
    return;
  }
  TORCH_CHECK(output.scalar_type() != kUInt16 && output.scalar_type() != kUInt32 && output.scalar_type() != kUInt64,
              "prod: not implemented for ",
              output.scalar_type(),
              " output on MPS");
  // dtype=: cast the input to the output dtype and recurse into the same-dtype
  // kernel rather than requesting an unregistered (in, out) pair.
  if (!prod_kernel_registered(input.scalar_type(), output.scalar_type())) {
    if (input.scalar_type() == output.scalar_type()) {
      TORCH_CHECK(false, "prod for ", output.scalar_type(), " is not implemented on MPS");
    }
    prod_kernel_mps(input.to(output.scalar_type()).contiguous(), dims, keepdim, output);
    return;
  }
  if (input.numel() == 0) {
    output.fill_(1);
    return;
  }
  if (output.numel() == 0) {
    return;
  }
  // The native kernel indexes with uint32; reject tensors that would overflow.
  TORCH_CHECK(
      input.numel() <= std::numeric_limits<uint32_t>::max() && output.numel() <= std::numeric_limits<uint32_t>::max(),
      "MPS prod: tensor too large for 32-bit indexing");
  TORCH_CHECK(static_cast<uint32_t>(input.dim()) <= c10::metal::max_ndim,
              "prod: tensor rank > ",
              c10::metal::max_ndim,
              " is not supported on MPS");
  // in_dtype == input's dtype (routing above already cast for dtype=), so
  // make_reduction does not re-cast; dispatch on the shared kernel.
  Tensor result = output;
  auto iter = make_reduction("prod", result, input, dims, keepdim, input.scalar_type(), output.scalar_type());
  reduction_dispatch_mps(iter,
                         ReductionDispatch{
                             .prefix = "prod_",
                             .input_kernel_dtype = input.scalar_type(),
                             .output_kernel_dtype = output.scalar_type(),
                             .partial_dtype = output.scalar_type(),
                             .pass2_prefix = "prod_",
                             .has_strided_pass1 = false,
                             .inner_specializations = true,
                         });
}

TORCH_IMPL_FUNC(prod_out_mps)
(const Tensor& input_t, int64_t dim, bool keepdim, std::optional<ScalarType> dtype, const Tensor& output_t) {
  int64_t dims[1] = {dim};
  prod_kernel_mps(input_t, IntArrayRef(dims, 1), keepdim, output_t);
}

static void aminmax_kernel_mps(const Tensor& self, int64_t dim, bool keepdim, Tensor& min, Tensor& max) {
  TORCH_CHECK(!c10::isComplexType(self.scalar_type()), "aminmax not implemented for ", self.scalar_type());
  at::amin_outf(self, IntArrayRef(&dim, 1), keepdim, min);
  at::amax_outf(self, IntArrayRef(&dim, 1), keepdim, max);
}

static void aminmax_allreduce_kernel_mps(const Tensor& self, Tensor& min, Tensor& max) {
  TORCH_CHECK(!c10::isComplexType(self.scalar_type()), "aminmax not implemented for ", self.scalar_type());
  at::amin_outf(self, IntArrayRef{}, /*keepdim=*/false, min);
  at::amax_outf(self, IntArrayRef{}, /*keepdim=*/false, max);
}

Tensor prod_mps(const Tensor& self, std::optional<ScalarType> opt_dtype) {
  std::vector<int64_t> dims(self.dim());
  std::iota(dims.begin(), dims.end(), 0);

  Tensor output_t =
      at::empty({}, get_dtype_from_self(self, opt_dtype, true), std::nullopt, kMPS, std::nullopt, std::nullopt);

  prod_kernel_mps(self, IntArrayRef(dims), false, output_t);

  return output_t;
}

Tensor count_nonzero_mps(const Tensor& self, IntArrayRef dims) {
  Tensor result = create_reduction_result(self, dims, /*keepdim=*/false, ScalarType::Long);
  auto iter =
      make_reduction("count_nonzero_mps", result, self, dims, /*keepdim=*/false, self.scalar_type(), ScalarType::Long);
  count_nonzero_kernel_mps(iter);
  return result;
}

Tensor var_mps(const Tensor& input_t,
               at::OptionalIntArrayRef dim,
               const std::optional<Scalar>& correction,
               bool keepdim) {
  return std_var_common_impl_mps(input_t, dim, correction, keepdim, STANDARD_VARIANCE);
}

Tensor std_mps(const Tensor& input_t,
               at::OptionalIntArrayRef dim,
               const std::optional<Scalar>& correction,
               bool keepdim) {
  return std_var_common_impl_mps(input_t, dim, correction, keepdim, STANDARD_DEVIATION);
}

//-----------------------------------------------------------------------
// Min and max functions

// Max entire tensor into scalar result
Tensor max_mps(const Tensor& input_t) {
  return min_max_mps_impl(input_t, MPSReductionType::MAX, "max_mps");
}

// Min entire tensor into scalar result
Tensor min_mps(const Tensor& input_t) {
  return min_max_mps_impl(input_t, MPSReductionType::MIN, "min_mps");
}

// Max out with dim
TORCH_IMPL_FUNC(max_out_mps)
(const Tensor& input_t, int64_t dim, bool keepdim, const Tensor& output_t, const Tensor& indices_t) {
  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  native::zero_numel_check_dims(input_t, dim_, "max()");

  min_max_out_mps(input_t, dim, keepdim, output_t, indices_t, MPSReductionType::MAX, "max_out_mps");
}

// Min out with dim
TORCH_IMPL_FUNC(min_out_mps)
(const Tensor& input_t, int64_t dim, bool keepdim, const Tensor& output_t, const Tensor& indices_t) {
  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  native::zero_numel_check_dims(input_t, dim_, "min()");

  min_max_out_mps(input_t, dim, keepdim, output_t, indices_t, MPSReductionType::MIN, "min_out_mps");
}

TORCH_IMPL_FUNC(argmax_out_mps)
(const Tensor& input_t, std::optional<int64_t> dim, bool keepdim, const Tensor& output_t) {
  argmax_argmin_out_mps(input_t, dim, keepdim, output_t, MPSReductionType::MAX, "argmax_out_mps");
}

TORCH_IMPL_FUNC(argmin_out_mps)
(const Tensor& input_t, std::optional<int64_t> dim, bool keepdim, const Tensor& output_t) {
  argmax_argmin_out_mps(input_t, dim, keepdim, output_t, MPSReductionType::MIN, "argmin_out_mps");
}

// Max with dim
static std::tuple<Tensor, Tensor> max_mps(const Tensor& input_t, int64_t dim, bool keepdim) {
  return min_max_mps_impl(input_t, dim, keepdim, MPSReductionType::MAX, "max_mps");
}

// Min with dim
static std::tuple<Tensor, Tensor> min_mps(const Tensor& input_t, int64_t dim, bool keepdim) {
  return min_max_mps_impl(input_t, dim, keepdim, MPSReductionType::MIN, "min_mps");
}

std::tuple<Tensor, Tensor> std_mean_mps(const Tensor& self,
                                        at::OptionalIntArrayRef dim,
                                        const std::optional<Scalar>& correction,
                                        bool keepdim) {
  if (self.dim() == 0) {
    if (dim.has_value()) {
      (void)at::dim_list_to_bitset(dim.value(), self.dim());
    }
    auto self_1d = self.unsqueeze(0);
    auto [s, m] = std_mean_mps(self_1d, IntArrayRef({0}), correction, false);
    return {s.squeeze(), m.squeeze()};
  }
  TORCH_CHECK(c10::isFloatingType(self.scalar_type()) || c10::isComplexType(self.scalar_type()),
              "std_mean only support floating point and complex dtypes");
  if (c10::isComplexType(self.scalar_type())) {
    auto re = at::real(self).contiguous();
    auto im = at::imag(self).contiguous();
    auto var = at::var(re, dim, correction, keepdim).add(at::var(im, dim, correction, keepdim));
    auto mean = at::complex(at::mean(re, dim, keepdim), at::mean(im, dim, keepdim));
    return {var.sqrt(), mean};
  }
  auto reduce_dims = get_reduce_dims(self, dim);
  const auto correction_value = correction.value_or(1.0).toDouble();

  std::vector<int64_t> output_shape;
  for (int64_t d = 0; d < self.dim(); d++) {
    bool reduced = false;
    for (auto rd : reduce_dims) {
      if (rd == d) {
        reduced = true;
        break;
      }
    }
    if (reduced) {
      if (keepdim)
        output_shape.push_back(1);
    } else {
      output_shape.push_back(self.size(d));
    }
  }

  auto std_out = at::empty(output_shape, self.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  auto mean_out =
      at::empty(output_shape, self.scalar_type(), std::nullopt, kMPS, std::nullopt, MemoryFormat::Contiguous);

  if (std_out.numel() > 0) {
    if (self.numel() == 0) {
      // std/mean of an empty reduction is NaN (matches CPU and the std-out path).
      std_out.fill_(std::numeric_limits<float>::quiet_NaN());
      mean_out.fill_(std::numeric_limits<float>::quiet_NaN());
    } else {
      welford_kernel_mps(self, reduce_dims, keepdim, correction_value, true, std_out, &mean_out);
    }
  }

  return {std_out, mean_out};
}

std::tuple<Tensor, Tensor> var_mean_mps(const Tensor& self,
                                        at::OptionalIntArrayRef dim,
                                        const std::optional<Scalar>& correction,
                                        bool keepdim) {
  if (self.dim() == 0) {
    if (dim.has_value()) {
      (void)at::dim_list_to_bitset(dim.value(), self.dim());
    }
    auto self_1d = self.unsqueeze(0);
    auto [v, m] = var_mean_mps(self_1d, IntArrayRef({0}), correction, false);
    return {v.squeeze(), m.squeeze()};
  }
  TORCH_CHECK(c10::isFloatingType(self.scalar_type()) || c10::isComplexType(self.scalar_type()),
              "var_mean only support floating point and complex dtypes");
  if (c10::isComplexType(self.scalar_type())) {
    auto re = at::real(self).contiguous();
    auto im = at::imag(self).contiguous();
    auto var = at::var(re, dim, correction, keepdim).add(at::var(im, dim, correction, keepdim));
    auto mean = at::complex(at::mean(re, dim, keepdim), at::mean(im, dim, keepdim));
    return {var, mean};
  }
  auto reduce_dims = get_reduce_dims(self, dim);
  const auto correction_value = correction.value_or(1.0).toDouble();

  std::vector<int64_t> output_shape;
  for (int64_t d = 0; d < self.dim(); d++) {
    bool reduced = false;
    for (auto rd : reduce_dims) {
      if (rd == d) {
        reduced = true;
        break;
      }
    }
    if (reduced) {
      if (keepdim)
        output_shape.push_back(1);
    } else {
      output_shape.push_back(self.size(d));
    }
  }

  auto var_out = at::empty(output_shape, self.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  auto mean_out =
      at::empty(output_shape, self.scalar_type(), std::nullopt, kMPS, std::nullopt, MemoryFormat::Contiguous);

  if (var_out.numel() > 0) {
    if (self.numel() == 0) {
      // var/mean of an empty reduction is NaN (matches CPU and the var-out path).
      var_out.fill_(std::numeric_limits<float>::quiet_NaN());
      mean_out.fill_(std::numeric_limits<float>::quiet_NaN());
    } else {
      welford_kernel_mps(self, reduce_dims, keepdim, correction_value, false, var_out, &mean_out);
    }
  }

  return {var_out, mean_out};
}

REGISTER_DISPATCH(norm_stub, &norm_kernel_mps)
REGISTER_DISPATCH(sum_stub, &sum_kernel_mps)
REGISTER_DISPATCH(nansum_stub, &nansum_kernel_mps)
REGISTER_DISPATCH(mean_stub, &mean_kernel_mps)
REGISTER_DISPATCH(min_values_stub, &min_values_kernel_mps)
REGISTER_DISPATCH(max_values_stub, &max_values_kernel_mps)
REGISTER_DISPATCH(and_stub, &and_kernel_mps)
REGISTER_DISPATCH(or_stub, &or_kernel_mps)
REGISTER_DISPATCH(aminmax_stub, &aminmax_kernel_mps)
REGISTER_DISPATCH(aminmax_allreduce_stub, &aminmax_allreduce_kernel_mps)

} // namespace at::native
