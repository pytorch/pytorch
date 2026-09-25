//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/CollapseDims.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/native/Pool.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/ReduceOps.h>
#include <c10/util/irange.h>
#include <algorithm>
#include <bit>
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
#include <ATen/ops/count_nonzero_native.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/max_native.h>
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

  TORCH_CHECK_NOT_IMPLEMENTED(canUse32BitIndexMath(input, 1LL << 32),
                              "MPS norm: tensors requiring 64-bit indexing are not supported (numel=",
                              input.numel(),
                              ")");
  // Number of input elements that are reduced into one output element
  uint32_t reduction_size = input.numel() / output.numel();

  TORCH_INTERNAL_ASSERT(output.dim() == input.dim());

  // Fast path: L1/L2 norm over the innermost contiguous dim reuses the sum
  // innermost kernel (abs/square load + sqrt)
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
      auto kernel_name = fmt::format("norm_{}_reduction_innermost_{}_{}",
                                     p == 2.0 ? "l2" : "l1",
                                     scalarToMetalTypeString(input),
                                     scalarToMetalTypeString(output));
      constexpr uint32_t rows_per_tg = INNER_TG_SIZE / c10::metal::simdgroup_size;
      const auto num_tgs = c10::metal::ceil_div(M, rows_per_tg);
      MPSStream* stream = getCurrentMPSStream();
      return dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          id<MTLComputeCommandEncoder> ce = stream->commandEncoder();
          auto ps = lib.getPipelineStateForFunc(kernel_name);
          getMPSProfiler().beginProfileKernel(ps, "norm_reduction_innermost", {input}, stream);
          [ce setComputePipelineState:ps];
          mtl_setArgs(ce, input, output, std::array<uint32_t, 2>{M, N}, 0.0f);
          [ce dispatchThreads:MTLSizeMake(num_tgs * INNER_TG_SIZE, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(INNER_TG_SIZE, 1, 1)];
          getMPSProfiler().endProfileKernel(ps, stream);
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
      getMPSProfiler().beginProfileKernel(pipeline_state, "norm", {input}, stream);
      [compute_encoder setComputePipelineState:pipeline_state];
      mtl_setArgs(compute_encoder, input, output, params);

      auto threads_per_group = std::min(MAX_THREADGROUP_SIZE, reduction_size);
      const auto num_threads = static_cast<uint64_t>(output.numel()) * threads_per_group;

      [compute_encoder dispatchThreads:MTLSizeMake(num_threads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];

      getMPSProfiler().endProfileKernel(pipeline_state, stream);
    }
  });
}

static void argmax_argmin_out_mps(const Tensor& input_t,
                                  std::optional<int64_t> dim,
                                  bool keepdim,
                                  const Tensor& output_t,
                                  MPSReductionType reduction_type,
                                  const std::string& func_name);
static void value_reduction_kernel_mps(TensorIterator& iter, const std::string& op_prefix);

static Tensor min_max_mps_impl(const Tensor& input_t, MPSReductionType reduction_type, const std::string& func_name) {
  Tensor output_t = at::empty({}, input_t.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  if (output_t.numel() == 0 || input_t.numel() == 0) {
    return output_t;
  }
  auto iter = at::meta::make_reduction(input_t, output_t, IntArrayRef{}, /*keepdim=*/false, input_t.scalar_type());
  value_reduction_kernel_mps(iter, reduction_type == MPSReductionType::MIN ? "min_" : "max_");
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

  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  argmax_argmin_out_mps(input_t, dim_, keepdim, indices_t, reduction_type, func_name);
  int64_t dims[1] = {dim_};
  auto iter = at::meta::make_reduction(input_t, output_t, IntArrayRef(dims, 1), keepdim, input_t.scalar_type());
  value_reduction_kernel_mps(iter, reduction_type == MPSReductionType::MIN ? "min_" : "max_");
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

// Kernels view the input as [outer, dim, inner] with the middle `dim` reduced:
// e.g. reducing dim 1 of a [2, 3, 4, 5] tensor gives outer=2, dim=3, inner=20
enum class ReductionKernel {
  Generic, // fallback for any layout, one threadgroup per output
  Innermost, // inner == 1, contiguous: one simdgroup per outer (row)
  InnermostChunk, // Innermost with dim <= 256, several rows per simdgroup
  Outer, // inner > 1: threadgroups of 32 columns x 32 threads splitting `dim`
  OuterSmallDim, // inner > 1 and short dim (<= 256): one thread reduces a whole column
  Narrow, // contiguous Outer with inner < 32: one threadgroup reduces all inner columns at once
  Flat, // large full reduction (single output): pass 1 over contiguous slices
  ArgCombine // argmax/argmin pass 2: merges segments, first occurrence wins
};

struct ReductionLayout {
  uint32_t outer_size;
  uint32_t dim_size;
  uint32_t inner_size;
  std::array<uint32_t, 4> strides;
  bool is_contiguous;

  static ReductionLayout contiguous(uint32_t outer_size, uint32_t dim_size, uint32_t inner_size) {
    return {outer_size, dim_size, inner_size, {inner_size, 1, dim_size * inner_size, 0}, true};
  }
};

struct ReductionPlan {
  ReductionKernel kernel = ReductionKernel::Generic;
  ReductionLayout layout;
  uint32_t num_segments = 1;
  uint32_t lanes = c10::metal::simdgroup_size;
};

static std::optional<ReductionLayout> outer_reduction_layout(const Tensor& input, int64_t reduced_dim) {
  c10::DimVector sizes(input.sizes().begin(), input.sizes().end());
  c10::DimVector strides(input.strides().begin(), input.strides().end());
  const auto [collapsed_dim, collapsed_ndim] =
      at::collapse_dims(sizes.data(), strides.data(), input.dim(), reduced_dim);
  // Usable when at most one collapsed block remains on each side of the
  // reduced dim: [dim], [dim, inner], [outer, dim] or [outer, dim, inner].
  if (collapsed_ndim > 2 && !(collapsed_ndim == 3 && collapsed_dim == 1)) {
    return std::nullopt;
  }
  const bool has_outer = collapsed_dim > 0;
  const bool has_inner = collapsed_dim < collapsed_ndim - 1;
  return ReductionLayout{has_outer ? safe_downcast<uint32_t, int64_t>(sizes[0]) : 1u,
                         safe_downcast<uint32_t, int64_t>(sizes[collapsed_dim]),
                         has_inner ? safe_downcast<uint32_t, int64_t>(sizes[collapsed_dim + 1]) : 1u,
                         {safe_downcast<uint32_t, int64_t>(strides[collapsed_dim]),
                          has_inner ? safe_downcast<uint32_t, int64_t>(strides[collapsed_dim + 1]) : 0u,
                          has_outer ? safe_downcast<uint32_t, int64_t>(strides[0]) : 0u,
                          0},
                         input.is_contiguous()};
}

static ReductionPlan select_outer_reduction(const ReductionLayout& layout, bool is_arg, int64_t numel) {
  const auto [outer_size, dim_size, inner_size, strides, is_contiguous] = layout;
  const auto natural_tgs = outer_size * at::ceil_div(inner_size, OUTER_TG_WIDTH);
  // Tall skinny case (few columns, long reduced dim): too few
  // threadgroups to fill the GPU, so split the reduced dim into segments
  // and fold the [num_segs, inner_size] partials in a second pass.
  const bool split = outer_size == 1 && dim_size >= OUTER_SPLIT_MIN_DIM_SIZE && natural_tgs < OUTER_SPLIT_MIN_TGS;
  // Short reduced dim: a 32-row threadgroup would idle most rows, so use
  // the small-dim layout (one thread walks the whole reduced dim) when
  // there are enough columns to fill the GPU with 32-wide threadgroups.
  const bool small_dim = !is_arg && dim_size <= OUTER_SMALL_DIM_MAX_SIZE && natural_tgs >= SPLIT_MIN_TGS;
  // inner_size narrower than a threadgroup row: the narrow layout is the
  // only one that keeps a full threadgroup busy. Tensors below
  // CHUNK_MIN_NUMEL are enqueue-bound and stay on the single-dispatch
  // outer kernel below. When the small-dim layout is available and the
  // reduced dim is short, its serial row walk beats narrow's mostly-idle
  // threadgroups.
  // argmax/argmin only have the narrow pass-1 layout, so they take it only
  // when splitting.
  const bool use_narrow =
      is_arg ? split : numel >= CHUNK_MIN_NUMEL && !(small_dim && dim_size < NARROW_BATCHED_MIN_DIM_SIZE);
  ReductionPlan plan{.kernel = ReductionKernel::Outer, .layout = layout};

  if (is_contiguous && inner_size < OUTER_TG_WIDTH && use_narrow) {
    plan.kernel = ReductionKernel::Narrow;
    // Narrow routing:
    // batched or single-threadgroup single dispatch, or split-K two-pass. A
    // single narrow threadgroup saturates at ~NARROW_SPLIT_ELEMS_PER_TG
    // elements; past that, split the reduced dim into segments reduced in
    // parallel and fold the [num_segs, inner_size] partials in a second pass.
    plan.num_segments = outer_size > 1
        ? 1u
        : std::clamp(dim_size * inner_size / NARROW_SPLIT_ELEMS_PER_TG, is_arg ? 2u : 1u, SPLIT_MAX_SEGS);
  } else if (small_dim) {
    plan.kernel = ReductionKernel::OuterSmallDim;
  } else if (split) {
    plan.num_segments = is_contiguous && !is_arg
        ? std::min(dim_size / OUTER_SPLIT_MIN_SEG_LEN, std::max(2u, OUTER_SPLIT_MAX_TGS / natural_tgs))
        : std::clamp(OUTER_SPLIT_STRIDED_TARGET_TGS / natural_tgs, 2u, std::min(dim_size, SPLIT_MAX_SEGS));
  }
  return plan;
}

static ReductionPlan select_inner_reduction(const ReductionLayout& layout, bool is_arg, int64_t numel) {
  const auto num_rows = layout.outer_size;
  const auto row_len = layout.dim_size;
  ReductionPlan plan{.kernel = ReductionKernel::Innermost, .layout = layout};

  // Tensors too small to fill the GPU gain nothing from packing rows
  // into simdgroups; keep them on the innermost kernel below (the pre-chunk
  // routing, whose enqueue floor measures ~10% lower there).
  if (!is_arg && row_len <= CHUNK_MAX_ROW_LEN && numel >= CHUNK_MIN_NUMEL) {
    plan.kernel = ReductionKernel::InnermostChunk;
    // Smallest power-of-two lane count keeping at most CHUNK_ELEMS_PER_LANE
    // elements per lane; the chunk kernel then packs simdgroup_size / lanes
    // rows into one simdgroup instead of letting a short row idle most lanes.
    plan.lanes = std::min(c10::metal::simdgroup_size, std::bit_ceil(at::ceil_div(row_len, CHUNK_ELEMS_PER_LANE)));
    return plan;
  }

  // Skinny-M/huge-K: one simdgroup per row would leave the GPU
  // under-occupied below SPLIT_MIN_TGS threadgroups, so split each
  // row into segments and fold the [num_rows, num_segs] partials in
  // pass 2.
  const auto num_tgs = at::ceil_div(num_rows, INNER_TG_SIZE / c10::metal::simdgroup_size);
  if (num_tgs < (is_arg ? ARG_SPLIT_MIN_TGS : SPLIT_MIN_TGS) && row_len >= SPLIT_MIN_ROW_LEN) {
    uint32_t segments;
    if (is_arg) {
      segments = std::clamp(ARG_SPLIT_TARGET_PARTIALS / num_rows, 2u, std::max(row_len / ARG_SPLIT_MIN_SEG_LEN, 2u));
    } else {
      plan.kernel = ReductionKernel::InnermostChunk;
      // Segments per row for split-K: aim for ~SPLIT_TARGET_PARTIALS partials
      // (rows * segments) so pass 1 fills the GPU, cap so a segment keeps
      // >= SPLIT_MIN_SEG_LEN elements, then round so the segments come out
      // equal-sized.
      const auto max_segments = std::min(SPLIT_MAX_SEGS, at::ceil_div(row_len, SPLIT_MIN_SEG_LEN));
      segments = std::clamp(at::ceil_div(SPLIT_TARGET_PARTIALS, num_rows), 2u, std::max(max_segments, 2u));
    }
    plan.num_segments = at::ceil_div(row_len, at::ceil_div(row_len, segments));
  }
  return plan;
}

static ReductionPlan select_reduction_plan(const Tensor& input, const Tensor& output, bool is_arg) {
  const auto reduction_size = safe_downcast<uint32_t, int64_t>(input.numel() / output.numel());
  const auto num_outputs = safe_downcast<uint32_t, int64_t>(output.numel());
  ReductionPlan plan{.layout = ReductionLayout::contiguous(num_outputs, reduction_size, 1)};

  // Two-pass for large full reductions: pass 1 splits input into <=512
  // contiguous slices, each TG reduces one slice to a partial; pass 2
  // collapses the num_groups partials into the final scalar.
  if (num_outputs == 1 && !is_arg) {
    auto num_groups = std::min(512u, at::ceil_div(reduction_size, MAX_THREADGROUP_SIZE * SUM_NCHAINS));
    while (num_groups > 1 && reduction_size % num_groups != 0) {
      num_groups--;
    }
    if (num_groups > 1) {
      plan.kernel = ReductionKernel::Flat;
      plan.num_segments = num_groups;
    }
    return plan;
  }
  // The outer and innermost kernels index in 32 bits.
  if (output.is_contiguous() && canUse32BitIndexMath(input)) {
    int num_reduced = 0;
    int64_t first_reduced = input.dim();
    int64_t reduced_dim = -1;
    for (const auto d : c10::irange(input.dim())) {
      if (input.size(d) != output.size(d)) {
        num_reduced++;
        first_reduced = std::min(first_reduced, d);
        reduced_dim = d;
      }
    }
    const bool reduced_dims_adjacent = num_reduced == (reduced_dim - first_reduced + 1);
    if (num_reduced == 1 && reduced_dim < input.dim() - 1) {
      if (auto layout = outer_reduction_layout(input, reduced_dim)) {
        return select_outer_reduction(*layout, is_arg, input.numel());
      }
    } else if (num_reduced <= 1 && input.is_contiguous()) {
      // Innermost dim, which also covers the flattened dim=None argmax/argmin
      // view. Strided innermost reductions stay on the generic kernel below.
      return select_inner_reduction(plan.layout, is_arg, input.numel());
    } else if (input.is_contiguous() && reduced_dims_adjacent) {
      return select_reduction_plan(
          input.flatten(first_reduced, reduced_dim), output.flatten(first_reduced, reduced_dim), is_arg);
    }
  }
  // Generic single-pass fallback.
  return plan;
}

static ReductionPlan reduction_combine_plan(const ReductionPlan& plan, bool is_arg) {
  if (is_arg) {
    return {
        .kernel = ReductionKernel::ArgCombine,
        .layout = ReductionLayout::contiguous(plan.layout.outer_size * plan.layout.inner_size, plan.num_segments, 1)};
  }
  auto kernel = plan.kernel;
  if (kernel == ReductionKernel::InnermostChunk) {
    kernel = ReductionKernel::Innermost;
  } else if (kernel == ReductionKernel::Flat) {
    kernel = ReductionKernel::Generic;
  }
  return {.kernel = kernel,
          .layout = ReductionLayout::contiguous(plan.layout.outer_size, plan.num_segments, plan.layout.inner_size)};
}

static const char* reduction_kernel_suffix(ReductionKernel kernel) {
  switch (kernel) {
    case ReductionKernel::Generic:
      return "";
    case ReductionKernel::Innermost:
      return "_innermost";
    case ReductionKernel::InnermostChunk:
      return "_innermost_chunk";
    case ReductionKernel::Outer:
      return "_outer";
    case ReductionKernel::OuterSmallDim:
      return "_outer_small_dim";
    case ReductionKernel::Narrow:
      return "_narrow";
    case ReductionKernel::Flat:
      return "_flat";
    case ReductionKernel::ArgCombine:
      return "_combine";
  }
  TORCH_INTERNAL_ASSERT(false, "Unknown reduction kernel");
}

struct MetalType {
  MetalType(ScalarType dtype) : name(scalarToMetalTypeString(dtype)), size(c10::elementSize(dtype)) {}
  MetalType(std::string name, size_t size) : name(std::move(name)), size(size) {}
  std::string name;
  size_t size;
};

struct ReductionOp {
  // True for argmax/argmin, which reduce to indices rather than values.
  bool is_arg;
  std::string prefix;
  MetalType input_type;
  MetalType output_type;
  float param = 0;
};

struct ReductionPartials {
  Tensor values;
  Tensor indices;
};

static void encode_reduction(MPSStream* stream,
                             const Tensor& input,
                             const Tensor& output,
                             const ReductionPlan& plan,
                             const ReductionOp& op,
                             const std::string& profile_name,
                             const ReductionPartials& partials = {}) {
  const auto& layout = plan.layout;
  const bool split_arg = op.is_arg && plan.num_segments > 1;
  const auto name = op.prefix + "reduction" + reduction_kernel_suffix(plan.kernel);
  std::string kernel_name;
  if (split_arg) {
    kernel_name = fmt::format("{}_p1_{}", name, op.input_type.name);
  } else if (plan.kernel == ReductionKernel::ArgCombine) {
    kernel_name = fmt::format("{}_{}", name, op.input_type.name);
  } else {
    kernel_name = fmt::format("{}_{}_{}", name, op.input_type.name, op.output_type.name);
  }
  auto encoder = stream->commandEncoder();
  auto pipeline = lib.getPipelineStateForFunc(kernel_name);
  getMPSProfiler().beginProfileKernel(pipeline, profile_name.empty() ? name : profile_name, {input}, stream);
  [encoder setComputePipelineState:pipeline];
  if (split_arg && plan.kernel != ReductionKernel::Narrow) {
    mtl_setArgs<4>(encoder, partials.values, partials.indices);
  } else if (plan.kernel == ReductionKernel::ArgCombine) {
    mtl_setArgs<3>(encoder, partials.indices);
  }

  MTLSize grid;
  MTLSize group;
  switch (plan.kernel) {
    case ReductionKernel::Innermost:
    case ReductionKernel::InnermostChunk:
    case ReductionKernel::ArgCombine: {
      uint32_t num_simdgroups = layout.outer_size * plan.num_segments;
      if (op.is_arg) {
        const std::array<uint32_t, 4> sizes{
            num_simdgroups, at::ceil_div(layout.dim_size, plan.num_segments), plan.num_segments, layout.dim_size};
        mtl_setArgs(encoder, input, output, sizes);
      } else if (plan.kernel == ReductionKernel::InnermostChunk) {
        const std::array<uint32_t, 4> sizes{layout.outer_size, layout.dim_size, plan.lanes, plan.num_segments};
        mtl_setArgs(encoder, input, output, sizes, op.param);
        num_simdgroups = at::ceil_div(num_simdgroups, c10::metal::simdgroup_size / plan.lanes);
      } else {
        const std::array<uint32_t, 2> sizes{layout.outer_size, layout.dim_size};
        mtl_setArgs(encoder, input, output, sizes, op.param);
      }
      const auto num_tgs = at::ceil_div(num_simdgroups, INNER_TG_SIZE / c10::metal::simdgroup_size);
      grid = MTLSizeMake(num_tgs * INNER_TG_SIZE, 1, 1);
      group = MTLSizeMake(INNER_TG_SIZE, 1, 1);
      break;
    }
    case ReductionKernel::Outer:
    case ReductionKernel::OuterSmallDim:
    case ReductionKernel::Narrow: {
      const bool narrow = plan.kernel == ReductionKernel::Narrow;
      const std::array<uint32_t, 4> sizes{layout.dim_size, layout.inner_size, plan.num_segments, 0};
      if (op.is_arg && narrow) {
        mtl_setArgs(encoder, input, partials.values, partials.indices, sizes);
      } else if (op.is_arg) {
        mtl_setArgs(encoder, input, output, sizes, layout.strides);
      } else {
        mtl_setArgs(encoder, input, output, sizes, op.param);
        if (!narrow) {
          mtl_setArgs<4>(encoder, layout.strides);
        }
      }
      if (narrow) {
        const auto active = (NARROW_TG_SIZE / layout.inner_size) * layout.inner_size;
        grid = MTLSizeMake(active, plan.num_segments, layout.outer_size);
        group = MTLSizeMake(active, 1, 1);
      } else {
        const auto height = plan.kernel == ReductionKernel::OuterSmallDim ? 1u : OUTER_TG_HEIGHT;
        const auto num_tgs = at::ceil_div(layout.inner_size, OUTER_TG_WIDTH);
        grid = MTLSizeMake(num_tgs * OUTER_TG_WIDTH, plan.num_segments * height, layout.outer_size);
        group = MTLSizeMake(OUTER_TG_WIDTH, height, 1);
      }
      break;
    }
    case ReductionKernel::Flat: {
      constexpr uint32_t TPG = 256;
      const std::array<uint32_t, 2> sizes{plan.num_segments, layout.dim_size / plan.num_segments};
      mtl_setArgs(encoder, input, output, sizes);
      grid = MTLSizeMake(plan.num_segments * TPG, 1, 1);
      group = MTLSizeMake(TPG, 1, 1);
      break;
    }
    case ReductionKernel::Generic: {
      NormParams params{};
      params.ndim = input.dim();
      params.p = op.param;
      params.reduction_size = layout.dim_size;
      for (const auto d : c10::irange(input.dim())) {
        params.input_sizes[d] = input.size(d);
        params.input_strides[d] = input.stride(d);
        params.output_sizes[d] = output.size(d);
        params.output_strides[d] = output.stride(d);
      }
      mtl_setArgs(encoder, input, output, params);
      // Round per-TG thread count up to a full simdgroup (32 lanes). With
      // fewer threads, inactive lanes still participate in simd_shuffle but
      // carry register-zero, corrupting min/max reductions whose identity
      // is not zero. Padding threads load Op::identity() and contribute
      // nothing to the result.
      const auto threads = std::min(MAX_THREADGROUP_SIZE, c10::metal::round_up(params.reduction_size, 32u));
      grid = MTLSizeMake(static_cast<uint64_t>(output.numel()) * threads, 1, 1);
      group = MTLSizeMake(threads, 1, 1);
      break;
    }
  }
  [encoder dispatchThreads:grid threadsPerThreadgroup:group];
  getMPSProfiler().endProfileKernel(pipeline, stream);
}

static void reduction_dispatch_mps(Tensor input,
                                   Tensor output,
                                   const ReductionOp& op,
                                   const MetalType& partial_type,
                                   const std::string& combine_prefix,
                                   const std::string& profile_name = {}) {
  TORCH_INTERNAL_ASSERT(input.numel() > 0 && output.numel() > 0);
  TORCH_INTERNAL_ASSERT(output.dim() == input.dim());
  TORCH_CHECK_NOT_IMPLEMENTED(canUse32BitIndexMath(input, 1LL << 32),
                              op.is_arg ? profile_name : "MPS " + op.prefix + "reduction",
                              ": tensors requiring 64-bit indexing are not supported (numel=",
                              input.numel(),
                              ")");
  // most fast kernels need a contiguous input. Transposed or permuted inputs
  // are contiguous in memory but have reordered dims. This restores the memory
  // order by sorting dims by stride and permute the output to match. Example:
  // `y = torch.randn(4, 8).t()` has sizes `[8, 4]` and strides `[1, 8]`,
  // so it isn't contiguous.
  // Sorting dims descendingly by stride gives `perm = [1, 0]`, and `y.permute(1, 0)` has sizes `[4, 8]`,
  // strides `[8, 1]`: contiguous. So `y.sum(1)` runs as a dim-0 sum over that view.
  // this is done so such cases do not fallback to slow(er) general kernel.
  if (!op.is_arg && !input.is_contiguous()) {
    c10::DimVector perm(input.dim());
    std::iota(perm.begin(), perm.end(), 0);
    std::ranges::stable_sort(perm, std::greater{}, [&](int64_t d) { return input.stride(d); });
    auto permuted = input.permute(perm);
    if (permuted.is_contiguous()) {
      input = std::move(permuted);
      output = output.permute(perm);
    }
  }

  const auto plan = select_reduction_plan(input, output, op.is_arg);
  auto stream = getCurrentMPSStream();
  // for 1 pass plans we need to just dispatch the reduction and return
  if (plan.num_segments == 1) {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        encode_reduction(stream, input, output, plan, op, profile_name);
      }
    });
    return;
  }
  // build pass 2 plan based on the pass 1 plan.
  const auto combine_plan = reduction_combine_plan(plan, op.is_arg);
  // Flat needs a contiguous input.
  if (plan.kernel == ReductionKernel::Flat) {
    input = input.contiguous();
  }
  ReductionPartials partials;
  // pass-1 output dtype: opmath of output.scalar_type() for sum
  // (fp16/bf16/chalf partials would round once per segment),
  // output.scalar_type() for min/max, uchar for all/any.
  const auto num_partials = output.numel() * plan.num_segments;
  partials.values = at::empty({num_partials * static_cast<int64_t>(partial_type.size)}, output.options().dtype(kByte));
  if (op.is_arg) {
    partials.indices = at::empty({num_partials}, output.options().dtype(kInt));
  }

  // Two-pass paths divide on the final pass only, while the accumulator is
  // still in opmath_t; sum and value kernels always take the param buffer, so
  // pass 1 binds a no-op 0 (arg kernels take none).
  const ReductionOp first_op{op.is_arg, op.prefix, op.input_type, partial_type};
  const ReductionOp combine_op{op.is_arg, combine_prefix, partial_type, op.output_type, op.param};
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      encode_reduction(stream, input, op.is_arg ? output : partials.values, plan, first_op, profile_name, partials);
      encode_reduction(stream, partials.values, output, combine_plan, combine_op, profile_name, partials);
    }
  });
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
  if (dim.has_value()) {
    input = input_t;
    output_view = keepdim ? output_t : output_t.unsqueeze(dim_);
    // A permuted view becomes contiguous once the reduced dim is moved
    // innermost or outermost (free for transposes); sliced or padded views
    // stay strided and are handled by the strided outer path below.
    // output_view moves along so the generic fallback still sees matching
    // input/output dim order.
    if (!input.is_contiguous()) {
      if (auto moved_inner = input_t.movedim(dim_, -1); moved_inner.is_contiguous()) {
        input = std::move(moved_inner);
        output_view = output_view.movedim(dim_, -1);
      } else if (auto moved_outer = input_t.movedim(dim_, 0); moved_outer.is_contiguous()) {
        input = std::move(moved_outer);
        output_view = output_view.movedim(dim_, 0);
      }
    }
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
  // Size-1 reduced dim: only index 0 is reachable.
  if (dim.has_value() && input_t.size(dim_) == 1) {
    output_t.fill_(0);
    return;
  }

  const ReductionOp op{/*is_arg=*/true, is_argmax ? "argmax_" : "argmin_", in_kdtype, kLong};
  // Winning values are input elements, so the value partials keep the input
  // dtype (no upcast needed) and the index partials are int32.
  reduction_dispatch_mps(input, output_view, op, in_kdtype, op.prefix, func_name);
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
  const ReductionOp op{/*is_arg=*/false, kernel_prefix, input.scalar_type(), output.scalar_type(), divisor};
  reduction_dispatch_mps(input, output, op, at::toOpMathType(output.scalar_type()), "sum_");
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
  const ReductionOp op{/*is_arg=*/false, op_prefix, in_kdtype, out_kdtype};
  reduction_dispatch_mps(input, output, op, partial_dtype, pass2_prefix);
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

TORCH_IMPL_FUNC(prod_out_mps)
(const Tensor& input_t, int64_t dim, bool keepdim, std::optional<ScalarType> dtype, const Tensor& output_t) {
  int64_t dims[1] = {dim};
  reduction_out_mps(input_t, IntArrayRef(dims, 1), keepdim, dtype, output_t, MPSReductionType::PROD, "prod_out_mps");
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

  reduction_out_mps(
      self, IntArrayRef(dims), false, opt_dtype, const_cast<Tensor&>(output_t), MPSReductionType::PROD, "prod_mps");

  return output_t;
}

Tensor count_nonzero_mps(const Tensor& self, IntArrayRef dims) {
  Tensor result = create_reduction_result(self, dims, /*keepdim=*/false, ScalarType::Long);
  auto iter =
      make_reduction("count_nonzero_mps", result, self, dims, /*keepdim=*/false, self.scalar_type(), ScalarType::Long);
  count_nonzero_kernel_mps(iter);
  return result;
}

static Tensor std_var_mps(const Tensor& self,
                          at::OptionalIntArrayRef dim,
                          const std::optional<Scalar>& correction,
                          bool keepdim,
                          bool take_sqrt) {
  TORCH_CHECK_TYPE(self.is_floating_point() || self.is_complex(),
                   "std and var only support floating point and complex dtypes");
  // Variance of a complex tensor is real: var(z) = var(Re z) + var(Im z).
  if (self.is_complex()) {
    auto var = std_var_mps(at::real(self), dim, correction, keepdim, /*take_sqrt=*/false);
    var.add_(std_var_mps(at::imag(self), dim, correction, keepdim, /*take_sqrt=*/false));
    return take_sqrt ? var.sqrt_() : var;
  }
  const auto dims = dim.value_or(IntArrayRef{});
  Tensor result = create_reduction_result(self, dims, keepdim, self.scalar_type());
  auto iter = make_reduction("std_var_mps", result, self, dims, keepdim, self.scalar_type(), self.scalar_type());
  if (self.numel() == 0) {
    return result.fill_(std::numeric_limits<float>::quiet_NaN());
  }
  if (result.numel() == 0) {
    return result;
  }
  const auto prefix = take_sqrt ? "std_" : "var_";
  const auto correction_value = static_cast<float>(correction.value_or(1).toDouble());
  const ReductionOp op{/*is_arg=*/false, prefix, self.scalar_type(), self.scalar_type(), correction_value};
  reduction_dispatch_mps(iter.input(0), iter.output(0), op, MetalType("float3", 16), prefix);
  return result;
}

Tensor var_mps(const Tensor& input_t,
               at::OptionalIntArrayRef dim,
               const std::optional<Scalar>& correction,
               bool keepdim) {
  return std_var_mps(input_t, dim, correction, keepdim, /*take_sqrt=*/false);
}

Tensor std_mps(const Tensor& input_t,
               at::OptionalIntArrayRef dim,
               const std::optional<Scalar>& correction,
               bool keepdim) {
  return std_var_mps(input_t, dim, correction, keepdim, /*take_sqrt=*/true);
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
  // TODO: Refactor it into a proper std_var_mean composite function
  auto std = std_mps(self, dim, correction, keepdim);
  auto mean = at::empty(std.sizes(), self.scalar_type(), std::nullopt, kMPS, std::nullopt, MemoryFormat::Contiguous);
  reduction_out_mps(self, dim, keepdim, std::nullopt, mean, MPSReductionType::MEAN, "mean_out_mps");
  return {std, mean};
}

std::tuple<Tensor, Tensor> var_mean_mps(const Tensor& self,
                                        at::OptionalIntArrayRef dim,
                                        const std::optional<Scalar>& correction,
                                        bool keepdim) {
  // TODO: Refactor it into a proper std_var_mean composite function
  auto var = var_mps(self, dim, correction, keepdim);
  auto mean = at::empty(var.sizes(), self.scalar_type(), std::nullopt, kMPS, std::nullopt, MemoryFormat::Contiguous);
  reduction_out_mps(self, dim, keepdim, std::nullopt, mean, MPSReductionType::MEAN, "mean_out_mps");
  return {var, mean};
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
