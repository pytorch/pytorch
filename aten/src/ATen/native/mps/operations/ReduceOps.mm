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
#include <ATen/ops/max_native.h>
#include <ATen/ops/mean_native.h>
#include <ATen/ops/min_native.h>
#include <ATen/ops/nansum_native.h>
#include <ATen/ops/prod_native.h>
#include <ATen/ops/std_mean_native.h>
#include <ATen/ops/std_native.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/sum_native.h>
#include <ATen/ops/trace_native.h>
#include <ATen/ops/var_mean_native.h>
#include <ATen/ops/var_native.h>
#include <ATen/ops/view_as_real.h>
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
      constexpr uint32_t rows_per_tg = INNER_TG_SIZE / c10::metal::simdgroup_size;
      const auto num_tgs = c10::metal::ceil_div(M, rows_per_tg);
      MPSStream* stream = getCurrentMPSStream();
      return dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          id<MTLComputeCommandEncoder> ce = stream->commandEncoder();
          auto ps = lib.getPipelineStateForFunc(kernel_name);
          getMPSProfiler().beginProfileKernel(ps, "norm_reduction_inner", {input}, stream);
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
      uint32_t num_threads = output.numel() * threads_per_group;

      [compute_encoder dispatchThreads:MTLSizeMake(num_threads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];

      getMPSProfiler().endProfileKernel(pipeline_state, stream);
    }
  });
}

static Tensor std_var_common_impl_mps(const Tensor& input_t,
                                      at::OptionalIntArrayRef dim,
                                      const std::optional<Scalar>& correction,
                                      bool keepdim,
                                      StdVarType stdVarType) {
  TORCH_CHECK_TYPE(input_t.is_floating_point() || input_t.is_complex(),
                   "std and var only support floating point and complex dtypes");

  // MPSGraph does not support complex dtypes. Decompose: var(z) = var(real) + var(imag).
  if (c10::isComplexType(input_t.scalar_type())) {
    auto real_view = at::view_as_real(input_t.contiguous()); // shape [..., 2]; last dim is [real, imag]
    auto real_part = real_view.select(-1, 0).contiguous();
    auto imag_part = real_view.select(-1, 1).contiguous();
    auto var_real = std_var_common_impl_mps(real_part, dim, correction, keepdim, STANDARD_VARIANCE);
    auto var_imag = std_var_common_impl_mps(imag_part, dim, correction, keepdim, STANDARD_VARIANCE);
    auto var_sum = var_real.add(var_imag);
    if (stdVarType == STANDARD_DEVIATION) {
      return var_sum.sqrt();
    }
    return var_sum;
  }

  using CachedGraph = MPSUnaryCachedGraph;

  IntArrayRef input_shape = input_t.sizes();
  int64_t num_input_dims = input_shape.size();

  bool use_dim = dim.has_value();
  IntArrayRef dim_value = use_dim ? dim.value() : NULL;

  if (use_dim) {
    std::string errMessage = (stdVarType == STANDARD_DEVIATION) ? "std_mps" : "var_mps";
    errMessage += ": reduction dim must be in the range of input shape";
    for (const auto dim : dim_value) {
      auto wrap_dim = maybe_wrap_dim(dim, num_input_dims);
      TORCH_CHECK(wrap_dim < (num_input_dims ? num_input_dims : 1), errMessage.c_str())
    }
  }

  bool use_correction = !(correction.has_value() && correction.value().toDouble() == 0);
  const auto correction_value = correction.value_or(1.0).toDouble();
  int64_t correction_n = 1;

  NSArray<NSNumber*>* wrappedAxes = getTensorAxes(input_t.sizes(), dim);

  int64_t num_output_dims = 0;
  NSMutableArray<NSNumber*>* axes = nil;
  NSMutableArray<NSNumber*>* apparent_output_shape = nil;
  NSMutableArray<NSNumber*>* apparent_input_shape = nil;
  std::vector<int64_t> output_shape;

  if ((!keepdim && !use_dim) || (!keepdim && use_dim && dim_value.size() <= 0)) {
    // Flatten the input tensor to reduce it to one value
    apparent_input_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    int64_t num_in_elements = c10::multiply_integers(input_shape);
    apparent_input_shape[0] = [NSNumber numberWithInt:num_in_elements];

    // Output is a single value
    apparent_output_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    apparent_output_shape[0] = @1;

    num_output_dims = 0;

    correction_n = num_in_elements;

    // Reduction axes
    axes = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    axes[0] = @0;
  } else if (!keepdim && use_dim && !dim_value.empty()) {
    int64_t num_reduce_dims = dim_value.size();
    num_output_dims = num_input_dims;

    set_axes(axes, num_reduce_dims, dim_value, num_input_dims);
    set_apparent_shapes(
        apparent_output_shape, apparent_input_shape, num_reduce_dims, num_output_dims, input_shape, axes);

    num_output_dims = (num_input_dims >= num_reduce_dims) ? (num_input_dims - num_reduce_dims) : 0; // num_input_dims;

    unsigned int curr_i = 0;
    for (const auto i : c10::irange(num_input_dims)) {
      bool found = false;
      for (const auto j : c10::irange(num_reduce_dims)) {
        if (i == maybe_wrap_dim(dim_value[j], num_input_dims)) {
          found = true;
          break;
        }
      }
      if (found) {
        continue;
      }
      output_shape.push_back(input_shape[i]);
      curr_i += 1;
      // End loop when output shape is filled
      if (curr_i == num_output_dims) {
        break;
      }
    }

    for (const auto dim : dim_value) {
      auto wrap_dim = maybe_wrap_dim(dim, input_shape.size());
      correction_n *= input_shape[wrap_dim];
    }
    // (3, 4, 5) --> (3, 5)
  } else if ((keepdim && !use_dim) || (keepdim && use_dim && dim_value.empty())) {
    num_output_dims = 0;
    int64_t num_reduce_dims = 0;
    set_axes(axes, num_reduce_dims, dim_value, input_shape.size());
    set_apparent_shapes(
        apparent_output_shape, apparent_input_shape, num_reduce_dims, num_output_dims, input_shape, axes);
    num_output_dims = num_input_dims;
    for (const auto i : c10::irange(num_input_dims)) {
      output_shape.push_back((int64_t)1);
      correction_n *= input_shape[i];
    }
    // scalar --> vector case [[1.0034567]]
  } else if (keepdim && use_dim && !dim_value.empty()) {
    int64_t num_reduce_dims = dim_value.size();
    num_output_dims = num_input_dims;

    set_axes(axes, num_reduce_dims, dim_value, num_input_dims);
    set_apparent_shapes(
        apparent_output_shape, apparent_input_shape, num_reduce_dims, num_output_dims, input_shape, axes);

    num_output_dims = num_input_dims; //(num_input_dims >= num_reduce_dims) ? (num_input_dims - num_reduce_dims) : 0;

    for (const int i : c10::irange(num_reduce_dims)) {
      auto wrap_dim = maybe_wrap_dim(dim_value[i], input_shape.size());
      correction_n *= input_shape[wrap_dim];
    }

    for (const int i : c10::irange(num_input_dims)) {
      output_shape.push_back([apparent_output_shape[i] longValue]);
    }
  }

  Tensor output_t = at::empty(IntArrayRef(output_shape.data(), num_output_dims),
                              input_t.scalar_type(),
                              std::nullopt,
                              kMPS,
                              std::nullopt,
                              std::nullopt);

  if (output_t.numel() == 0 || input_t.numel() == 0) {
    output_t.fill_(std::numeric_limits<float>::quiet_NaN());
    return output_t;
  }

  double dof = std::max(0.0, correction_n - correction_value);
  double bessel_correction = correction_n / dof;
  auto stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string op_key = (stdVarType == STANDARD_DEVIATION) ? "std_mps" : "var_mps";
    NSString* ns_key = [[wrappedAxes valueForKey:@"description"] componentsJoinedByString:@","];
    std::string bessel_corrected = (use_correction && correction_value) ? "unbiased " : "biased ";
    std::string use_dim_info = (use_dim) ? "use_dim=1:" + std::to_string(dim_value.size()) : "use_dim=0";
    std::string keepdim_info = (keepdim) ? "keepdim=1" : "keepdim=0";
    std::string key = op_key + ":" + getTensorsStringKey(input_t) + ":" + use_dim_info + ":" + keepdim_info + ":" +
        std::string([ns_key UTF8String]) + ":" + bessel_corrected + ":" + std::to_string(correction_value);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_t);
      MPSGraphTensor* outputVarTensor = [mpsGraph varianceOfTensor:inputTensor axes:wrappedAxes name:nil];
      MPSGraphTensor* outputTensor = nil;

      if (use_correction && correction_value) {
        MPSGraphTensor* besselTensor = [mpsGraph constantWithScalar:bessel_correction dataType:getMPSDataType(input_t)];
        MPSGraphTensor* correctedTensor = [mpsGraph multiplicationWithPrimaryTensor:outputVarTensor
                                                                    secondaryTensor:besselTensor
                                                                               name:nil];
        outputTensor = (stdVarType == STANDARD_DEVIATION) ? [mpsGraph squareRootWithTensor:correctedTensor name:nil]
                                                          : correctedTensor;
      } else {
        outputTensor = (stdVarType == STANDARD_DEVIATION) ? [mpsGraph squareRootWithTensor:outputVarTensor name:nil]
                                                          : outputVarTensor;
      }
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t, apparent_output_shape);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return output_t;
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
    // A permuted view becomes contiguous once the reduced dim is moved
    // innermost or outermost (free for transposes); sliced or padded views
    // stay strided and are handled by the strided outer path below.
    // output_view moves along so the generic fallback still sees matching
    // input/output dim order.
    if (!input.is_contiguous()) {
      if (auto moved_inner = input_t.movedim(dim_, -1); moved_inner.is_contiguous()) {
        input = std::move(moved_inner);
        reduce_dim = input.dim() - 1;
        output_view = output_view.movedim(dim_, -1);
      } else if (auto moved_outer = input_t.movedim(dim_, 0); moved_outer.is_contiguous()) {
        input = std::move(moved_outer);
        reduce_dim = 0;
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
  const auto op_prefix = is_argmax ? "argmax" : "argmin";
  const auto in_str = scalarToMetalTypeString(in_kdtype);
  MPSStream* stream = getCurrentMPSStream();
  const auto nd = input.dim();

  // Size-1 reduced dim: only index 0 is reachable.
  if (dim.has_value() && input.size(reduce_dim) == 1) {
    output_t.fill_(0);
    return;
  }

  auto encode_arg_inner = [&](const Tensor& in, uint32_t num_rows, uint32_t row_len) {
    const auto kname = fmt::format("{}_reduction_inner_{}_long", op_prefix, in_str);
    const auto num_tgs = at::ceil_div(num_rows, INNER_TG_SIZE / c10::metal::simdgroup_size);
    auto ce = stream->commandEncoder();
    auto ps = lib.getPipelineStateForFunc(kname);
    getMPSProfiler().beginProfileKernel(ps, func_name, {in}, stream);
    [ce setComputePipelineState:ps];
    const std::array<uint32_t, 4> sizes_s{num_rows, row_len, 0, 0};
    mtl_setArgs(ce, in, output_t, sizes_s);
    [ce dispatchThreads:MTLSizeMake(num_tgs * INNER_TG_SIZE, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(INNER_TG_SIZE, 1, 1)];
    getMPSProfiler().endProfileKernel(ps, stream);
  };
  // The outer kernel views the input as [outer_size, dim_size, inner_size]
  // through explicit strides (contiguous callers pass the contiguous
  // strides of that view), reducing dim; grid z walks the outer batches.
  // With partials, this is split-K pass 1 over a single batch: grid y cuts
  // the dim rows into num_segs segments, one (value, index) pair each.
  auto encode_arg_outer = [&](const Tensor& in,
                              uint32_t dim_size,
                              uint32_t inner_size,
                              uint32_t outer_size,
                              uint32_t dim_stride,
                              uint32_t inner_stride,
                              uint32_t outer_stride,
                              uint32_t num_segs,
                              const std::optional<std::pair<Tensor, Tensor>>& partials) {
    const auto split = partials.has_value();
    const auto kname = split ? fmt::format("{}_reduction_outer_p1_{}", op_prefix, in_str)
                             : fmt::format("{}_reduction_outer_{}_long", op_prefix, in_str);
    const auto num_tg_x = c10::metal::ceil_div(inner_size, OUTER_TG_WIDTH);
    auto ce = stream->commandEncoder();
    auto ps = lib.getPipelineStateForFunc(kname);
    getMPSProfiler().beginProfileKernel(ps, func_name, {in}, stream);
    [ce setComputePipelineState:ps];
    const std::array<uint32_t, 4> sizes_s{dim_size, inner_size, num_segs, 0};
    const std::array<uint32_t, 4> strides_s{dim_stride, inner_stride, outer_stride, 0};
    mtl_setArgs(ce, in, output_t, sizes_s, strides_s);
    if (split) {
      mtl_setArgs<4>(ce, partials->first, partials->second);
    }
    [ce dispatchThreads:MTLSizeMake(
                            num_tg_x * OUTER_TG_WIDTH, (split ? num_segs : 1) * OUTER_TG_HEIGHT, split ? 1 : outer_size)
        threadsPerThreadgroup:MTLSizeMake(OUTER_TG_WIDTH, OUTER_TG_HEIGHT, 1)];
    getMPSProfiler().endProfileKernel(ps, stream);
  };
  auto encode_arg_narrow_p1 = [&](const Tensor& in,
                                  const Tensor& vals,
                                  const Tensor& idxs,
                                  uint32_t dim_size,
                                  uint32_t inner_size,
                                  uint32_t num_segs) {
    const auto kname = fmt::format("{}_reduction_narrow_p1_{}", op_prefix, in_str);
    auto ce = stream->commandEncoder();
    auto ps = lib.getPipelineStateForFunc(kname);
    getMPSProfiler().beginProfileKernel(ps, func_name, {in}, stream);
    [ce setComputePipelineState:ps];
    const std::array<uint32_t, 4> sizes_s{dim_size, inner_size, num_segs, 0};
    mtl_setArgs(ce, in, vals, idxs, sizes_s);
    const auto active = (NARROW_TG_SIZE / inner_size) * inner_size;
    [ce dispatchThreads:MTLSizeMake(active, num_segs, 1) threadsPerThreadgroup:MTLSizeMake(active, 1, 1)];
    getMPSProfiler().endProfileKernel(ps, stream);
  };
  auto encode_arg_inner_p1 = [&](const Tensor& in,
                                 const Tensor& vals,
                                 const Tensor& idxs,
                                 uint32_t num_rows,
                                 uint32_t row_len,
                                 uint32_t seg_len,
                                 uint32_t num_segs) {
    const auto kname = fmt::format("{}_reduction_inner_p1_{}", op_prefix, in_str);
    const auto num_partials = num_rows * num_segs;
    const auto num_tgs = at::ceil_div(num_partials, INNER_TG_SIZE / c10::metal::simdgroup_size);
    auto ce = stream->commandEncoder();
    auto ps = lib.getPipelineStateForFunc(kname);
    getMPSProfiler().beginProfileKernel(ps, func_name, {in}, stream);
    [ce setComputePipelineState:ps];
    const std::array<uint32_t, 4> sizes_s{num_partials, seg_len, num_segs, row_len};
    mtl_setArgs(ce, in, output_t, sizes_s);
    mtl_setArgs<4>(ce, vals, idxs);
    [ce dispatchThreads:MTLSizeMake(num_tgs * INNER_TG_SIZE, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(INNER_TG_SIZE, 1, 1)];
    getMPSProfiler().endProfileKernel(ps, stream);
  };
  auto encode_arg_combine = [&](const Tensor& vals, const Tensor& idxs, uint32_t num_outputs, uint32_t num_segs) {
    const auto kname = fmt::format("{}_reduction_combine_{}", op_prefix, in_str);
    const auto num_tgs = at::ceil_div(num_outputs, INNER_TG_SIZE / c10::metal::simdgroup_size);
    auto ce = stream->commandEncoder();
    auto ps = lib.getPipelineStateForFunc(kname);
    getMPSProfiler().beginProfileKernel(ps, func_name, {vals}, stream);
    [ce setComputePipelineState:ps];
    const std::array<uint32_t, 4> sizes_s{num_outputs, num_segs, 0, 0};
    mtl_setArgs(ce, vals, output_t, sizes_s, idxs);
    [ce dispatchThreads:MTLSizeMake(num_tgs * INNER_TG_SIZE, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(INNER_TG_SIZE, 1, 1)];
    getMPSProfiler().endProfileKernel(ps, stream);
  };
  // Shared split-K driver: allocate the [num_outputs, num_segs] (value,
  // index) partials, run the layout-specific pass 1, resolve with combine.
  // Winning values are input elements, so the value partials keep the input
  // dtype (no upcast needed) and the index partials are int32.
  auto run_arg_split = [&](uint32_t num_outputs, uint32_t num_segs, void (^pass1)(const Tensor&, const Tensor&)) {
    auto val_partials = at::empty({(int64_t)num_outputs, (int64_t)num_segs}, input.options().dtype(in_kdtype));
    auto idx_partials = at::empty({(int64_t)num_outputs, (int64_t)num_segs}, input.options().dtype(kInt));
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        pass1(val_partials, idx_partials);
        encode_arg_combine(val_partials, idx_partials, num_outputs, num_segs);
      }
    });
  };

  // Non-contiguous input that movedim could not normalize but whose dims
  // still collapse to [outer_size, dim_size, inner_size] with the dim not
  // innermost (a sliced or padded view): the outer kernel indexes through
  // explicit strides. Strided innermost reductions stay on the generic
  // kernel below. All fast-path kernels index in 32 bits.
  if (!input.is_contiguous() && output_t.is_contiguous() && reduce_dim < nd - 1 && canUse32BitIndexMath(input)) {
    c10::DimVector sizes(input.sizes().begin(), input.sizes().end());
    c10::DimVector strides(input.strides().begin(), input.strides().end());
    const auto [collapsed_dim, collapsed_ndim] = at::collapse_dims(sizes.data(), strides.data(), nd, reduce_dim);
    // Usable when at most one collapsed block remains on each side of the
    // reduced dim: [dim], [dim, inner], [outer, dim] or [outer, dim, inner].
    if (collapsed_ndim <= 2 || (collapsed_ndim == 3 && collapsed_dim == 1)) {
      const auto has_outer = collapsed_dim > 0;
      const auto has_inner = collapsed_dim < collapsed_ndim - 1;
      const auto outer_size = has_outer ? safe_downcast<uint32_t, int64_t>(sizes[0]) : 1u;
      const auto dim_size = safe_downcast<uint32_t, int64_t>(sizes[collapsed_dim]);
      const auto inner_size = has_inner ? safe_downcast<uint32_t, int64_t>(sizes[collapsed_dim + 1]) : 1u;
      const auto dim_stride = safe_downcast<uint32_t, int64_t>(strides[collapsed_dim]);
      const auto inner_stride = has_inner ? safe_downcast<uint32_t, int64_t>(strides[collapsed_dim + 1]) : 0u;
      const auto outer_stride = has_outer ? safe_downcast<uint32_t, int64_t>(strides[0]) : 0u;
      const auto natural_tgs = outer_size * c10::metal::ceil_div(inner_size, OUTER_TG_WIDTH);
      // Tall skinny case: too few threadgroups to fill the GPU, so split
      // the reduced dim into segments and resolve the (value, index)
      // partials in a second pass.
      if (outer_size == 1 && dim_size >= OUTER_SPLIT_MIN_DIM_SIZE && natural_tgs < OUTER_SPLIT_MIN_TGS) {
        const auto num_segs =
            std::clamp(OUTER_SPLIT_STRIDED_TARGET_TGS / natural_tgs, 2u, std::min(dim_size, SPLIT_MAX_SEGS));
        run_arg_split(inner_size, num_segs, ^(const Tensor& vals, const Tensor& idxs) {
          encode_arg_outer(input, dim_size, inner_size, 1, dim_stride, inner_stride, 0, num_segs, {{vals, idxs}});
        });
        return;
      }
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          encode_arg_outer(
              input, dim_size, inner_size, outer_size, dim_stride, inner_stride, outer_stride, 1, std::nullopt);
        }
      });
      return;
    }
  }

  if (input.is_contiguous() && output_t.is_contiguous() && canUse32BitIndexMath(input)) {
    // Any non-innermost dim reduces through the outer layout, viewed as
    // [outer_size, dim_size, inner_size] with the dim reduced.
    if (reduce_dim < nd - 1) {
      const auto outer_size =
          safe_downcast<uint32_t, int64_t>(c10::multiply_integers(input.sizes().slice(0, reduce_dim)));
      const auto dim_size = safe_downcast<uint32_t, int64_t>(input.size(reduce_dim));
      const auto inner_size =
          safe_downcast<uint32_t, int64_t>(input.numel() / (static_cast<int64_t>(outer_size) * dim_size));
      const auto natural_tgs = outer_size * c10::metal::ceil_div(inner_size, OUTER_TG_WIDTH);
      if (outer_size == 1 && dim_size >= OUTER_SPLIT_MIN_DIM_SIZE && natural_tgs < OUTER_SPLIT_MIN_TGS) {
        // inner_size below a threadgroup row: the narrow pass-1 layout
        // keeps a full threadgroup busy.
        if (inner_size < OUTER_TG_WIDTH) {
          const auto num_segs = std::clamp<uint32_t>(
              (static_cast<int64_t>(dim_size) * inner_size) / NARROW_SPLIT_ELEMS_PER_TG, 2u, SPLIT_MAX_SEGS);
          run_arg_split(inner_size, num_segs, ^(const Tensor& vals, const Tensor& idxs) {
            encode_arg_narrow_p1(input, vals, idxs, dim_size, inner_size, num_segs);
          });
          return;
        }
        const auto num_segs =
            std::clamp(OUTER_SPLIT_STRIDED_TARGET_TGS / natural_tgs, 2u, std::min(dim_size, SPLIT_MAX_SEGS));
        run_arg_split(inner_size, num_segs, ^(const Tensor& vals, const Tensor& idxs) {
          encode_arg_outer(input, dim_size, inner_size, 1, inner_size, 1, 0, num_segs, {{vals, idxs}});
        });
        return;
      }
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          encode_arg_outer(
              input, dim_size, inner_size, outer_size, inner_size, 1, dim_size * inner_size, 1, std::nullopt);
        }
      });
      return;
    }
    // Innermost dim, which also covers the flattened dim=None view.
    // Skinny-M/huge-K inputs split each row into segments resolved in a
    // second pass.
    const auto row_len = safe_downcast<uint32_t, int64_t>(input.size(nd - 1));
    const auto num_rows = safe_downcast<uint32_t, int64_t>(input.numel() / row_len);
    const auto inner_tgs = at::ceil_div(num_rows, INNER_TG_SIZE / c10::metal::simdgroup_size);
    if (inner_tgs < ARG_SPLIT_MIN_TGS && row_len >= SPLIT_MIN_ROW_LEN) {
      const auto max_segs =
          std::min(row_len / ARG_SPLIT_MIN_SEG_LEN, std::max(2u, ARG_SPLIT_TARGET_PARTIALS / std::max(num_rows, 1u)));
      const auto seg_len = at::ceil_div(row_len, std::max(max_segs, 2u));
      const auto num_segs = at::ceil_div(row_len, seg_len);
      run_arg_split(num_rows, num_segs, ^(const Tensor& vals, const Tensor& idxs) {
        encode_arg_inner_p1(input, vals, idxs, num_rows, row_len, seg_len, num_segs);
      });
      return;
    }
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        encode_arg_inner(input, num_rows, row_len);
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
      getMPSProfiler().beginProfileKernel(ps, func_name, {input}, stream);
      [ce setComputePipelineState:ps];
      mtl_setArgs(ce, input, output_view, params);
      // Pad per-TG thread count up to a full simdgroup; padding lanes load
      // Op::identity() and skip the per-thread scan, keeping the two-stage
      // SIMD reduction well-defined for all reduction sizes.
      const auto threads_per_group = std::min(MAX_THREADGROUP_SIZE, c10::metal::round_up(params.reduction_size, 32u));
      const auto num_threads = static_cast<uint32_t>(output_view.numel()) * threads_per_group;
      [ce dispatchThreads:MTLSizeMake(num_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
      getMPSProfiler().endProfileKernel(ps, stream);
    }
  });
}

// Unified host-side dispatch for value-preserving reductions on MPS, shared
// by sum/nansum/mean/count_nonzero and min/max/all/any. Kernel name pattern
// is always `{prefix}reduction_{variant}_{TI}_{TO}` with variant in
// `""/"outer"/"outer_small_dim"/"narrow"/"narrow_strided"/"inner"/
// "inner_chunk"/"flat"/"strided"`; every op/dtype pair with a base kernel
// also has every variant (narrow_strided is sum-family only). Selects among
// these code paths:
//   1. Non-innermost dim reduction, viewed as [outer_size, dim_size,
//      inner_size] with the dim reduced (the CUDA spatial softmax / scan
//      decomposition): narrow for inner_size below a simdgroup,
//      outer_small_dim for short dim_size, split-K two-pass for tall skinny
//      inputs, outer otherwise. Taken for contiguous inputs and for
//      non-contiguous ones whose dims collapse to that view.
//   2. Last-dim reduction on contiguous input: inner_chunk for short rows,
//      split-K two-pass for skinny-M/huge-K, inner otherwise.
//   3. Two-pass full reduction (scalar output, large input).
//   4. Generic single-pass fallback.
struct ReductionDispatch {
  std::string prefix; // "sum_", "nansum_", "count_nonzero_", "min_", "max_",
                      // "all_", "any_".
  ScalarType input_kernel_dtype; // may differ from input.scalar_type() (e.g.
                                 // bool -> char for min/max).
  ScalarType output_kernel_dtype; // may differ from output.scalar_type() for
                                  // the same remap reason.
  ScalarType partial_dtype; // pass-1 output dtype: opmath of
                            // output.scalar_type() for sum (fp16/bf16/chalf
                            // partials would round once per segment),
                            // output.scalar_type() for min/max, uchar for
                            // all/any.
  std::string pass2_prefix; // pass-2 op prefix. count_nonzero -> "sum_" (the
                            // partials are already per-block counts), all/any
                            // -> "min_"/"max_" (predicate ran in pass 1).
  bool has_strided_pass1 = false; // sum has a `_strided_` pass-1 kernel; ops
                                  // without it call .contiguous() first.
  std::optional<float> divisor; // sum/mean only; appended as a float buffer
                                // to the outer/inner kernel signatures, and
                                // passed via NormParams.p elsewhere.
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

  auto encode_inner = [&](const Tensor& in,
                          const Tensor& out,
                          uint32_t num_rows,
                          uint32_t row_len,
                          const std::string& prefix,
                          ScalarType in_dt,
                          ScalarType out_dt,
                          std::optional<float> divisor) {
    auto kname =
        fmt::format("{}reduction_inner_{}_{}", prefix, scalarToMetalTypeString(in_dt), scalarToMetalTypeString(out_dt));
    const auto num_tgs = at::ceil_div(num_rows, INNER_TG_SIZE / c10::metal::simdgroup_size);
    id<MTLComputeCommandEncoder> ce = stream->commandEncoder();
    auto ps = lib.getPipelineStateForFunc(kname);
    getMPSProfiler().beginProfileKernel(ps, prefix + "reduction_inner", {in}, stream);
    [ce setComputePipelineState:ps];
    const std::array<uint32_t, 2> sizes_s{num_rows, row_len};
    mtl_setArgs(ce, in, out, sizes_s);
    if (divisor.has_value()) {
      mtl_setArgs<3>(ce, *divisor);
    }
    [ce dispatchThreads:MTLSizeMake(num_tgs * INNER_TG_SIZE, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(INNER_TG_SIZE, 1, 1)];
    getMPSProfiler().endProfileKernel(ps, stream);
  };
  // The outer kernels view the input as [outer_size, dim_size, inner_size]
  // with the dim reduced: each threadgroup covers OUTER_TG_WIDTH consecutive
  // inner columns, its OUTER_TG_HEIGHT rows walking the reduced dim; grid y
  // carries the split-K segments, grid z the outer batches. outer_small_dim
  // is the height-1 variant for short dim_size, where a full-height
  // threadgroup would mostly idle.
  auto encode_outer_strided = [&](const Tensor& in,
                                  const Tensor& out,
                                  uint32_t dim_size,
                                  uint32_t inner_size,
                                  uint32_t num_segs,
                                  uint32_t outer_size,
                                  bool is_small_dim,
                                  const std::string& prefix,
                                  ScalarType in_dt,
                                  ScalarType out_dt,
                                  std::optional<float> divisor,
                                  uint32_t dim_stride,
                                  uint32_t inner_stride,
                                  uint32_t outer_stride) {
    const auto tg_height = is_small_dim ? 1u : OUTER_TG_HEIGHT;
    auto kname = fmt::format("{}reduction_{}_{}_{}",
                             prefix,
                             is_small_dim ? "outer_small_dim" : "outer",
                             scalarToMetalTypeString(in_dt),
                             scalarToMetalTypeString(out_dt));
    const auto num_tg_x = c10::metal::ceil_div(inner_size, OUTER_TG_WIDTH);
    auto ce = stream->commandEncoder();
    auto ps = lib.getPipelineStateForFunc(kname);
    getMPSProfiler().beginProfileKernel(
        ps, prefix + (is_small_dim ? "reduction_outer_small_dim" : "reduction_outer"), {in}, stream);
    [ce setComputePipelineState:ps];
    const std::array<uint32_t, 4> sizes_s{dim_size, inner_size, 1, num_segs};
    const std::array<uint32_t, 4> strides_s{dim_stride, inner_stride, outer_stride, 0};
    mtl_setArgs(ce, in, out, sizes_s);
    if (divisor.has_value()) {
      mtl_setArgs<3>(ce, *divisor, strides_s);
    } else {
      mtl_setArgs<3>(ce, strides_s);
    }
    [ce dispatchThreads:MTLSizeMake(num_tg_x * OUTER_TG_WIDTH, num_segs * tg_height, outer_size)
        threadsPerThreadgroup:MTLSizeMake(OUTER_TG_WIDTH, tg_height, 1)];
    getMPSProfiler().endProfileKernel(ps, stream);
  };
  auto encode_outer = [&](const Tensor& in,
                          const Tensor& out,
                          uint32_t dim_size,
                          uint32_t inner_size,
                          uint32_t num_segs,
                          uint32_t outer_size,
                          const std::string& prefix,
                          ScalarType in_dt,
                          ScalarType out_dt,
                          std::optional<float> divisor) {
    encode_outer_strided(in,
                         out,
                         dim_size,
                         inner_size,
                         num_segs,
                         outer_size,
                         false,
                         prefix,
                         in_dt,
                         out_dt,
                         divisor,
                         inner_size,
                         1,
                         dim_size * inner_size);
  };
  auto encode_outer_small_dim = [&](const Tensor& in,
                                    const Tensor& out,
                                    uint32_t dim_size,
                                    uint32_t inner_size,
                                    uint32_t outer_size,
                                    const std::string& prefix,
                                    ScalarType in_dt,
                                    ScalarType out_dt,
                                    std::optional<float> divisor) {
    encode_outer_strided(in,
                         out,
                         dim_size,
                         inner_size,
                         1,
                         outer_size,
                         true,
                         prefix,
                         in_dt,
                         out_dt,
                         divisor,
                         inner_size,
                         1,
                         dim_size * inner_size);
  };
  // Narrow (contiguous or strided) handles inner_size < simdgroup_size, where
  // even the small-dim layout idles most of a threadgroup row; dispatched with
  // the largest multiple of inner_size <= NARROW_TG_SIZE threads.
  auto encode_narrow = [&](const Tensor& in,
                           const Tensor& out,
                           uint32_t dim_size,
                           uint32_t inner_size,
                           uint32_t num_segs,
                           uint32_t outer_size,
                           const std::string& prefix,
                           ScalarType in_dt,
                           ScalarType out_dt,
                           std::optional<float> divisor,
                           std::optional<std::array<uint32_t, 4>> strides) {
    const auto* variant = strides.has_value() ? "narrow_strided" : "narrow";
    auto kname = fmt::format(
        "{}reduction_{}_{}_{}", prefix, variant, scalarToMetalTypeString(in_dt), scalarToMetalTypeString(out_dt));
    auto ce = stream->commandEncoder();
    auto ps = lib.getPipelineStateForFunc(kname);
    getMPSProfiler().beginProfileKernel(ps, fmt::format("{}reduction_{}", prefix, variant), {in}, stream);
    [ce setComputePipelineState:ps];
    const std::array<uint32_t, 4> sizes_s{dim_size, inner_size, 1, num_segs};
    mtl_setArgs(ce, in, out, sizes_s);
    if (divisor.has_value()) {
      mtl_setArgs<3>(ce, *divisor);
    }
    if (strides.has_value()) {
      mtl_setArgs<4>(ce, *strides);
    }
    const auto active = (NARROW_TG_SIZE / inner_size) * inner_size;
    [ce dispatchThreads:MTLSizeMake(active, num_segs, outer_size) threadsPerThreadgroup:MTLSizeMake(active, 1, 1)];
    getMPSProfiler().endProfileKernel(ps, stream);
  };
  // Smallest power-of-two lane count keeping at most CHUNK_ELEMS_PER_LANE
  // elements per lane; the chunk kernel then packs simdgroup_size / lanes
  // rows into one simdgroup instead of letting a short row idle most lanes.
  auto pick_lanes = [](uint32_t row_len) -> uint32_t {
    return std::min(c10::metal::simdgroup_size, std::bit_ceil(at::ceil_div(row_len, CHUNK_ELEMS_PER_LANE)));
  };
  auto encode_chunk = [&](const Tensor& in,
                          const Tensor& out,
                          uint32_t num_rows,
                          uint32_t row_len,
                          uint32_t lanes,
                          uint32_t segments,
                          const std::string& prefix,
                          ScalarType in_dt,
                          ScalarType out_dt,
                          std::optional<float> divisor) {
    auto kname = fmt::format(
        "{}reduction_inner_chunk_{}_{}", prefix, scalarToMetalTypeString(in_dt), scalarToMetalTypeString(out_dt));
    const auto rows_per_simd = c10::metal::simdgroup_size / lanes;
    const auto total_simds = at::ceil_div(num_rows * segments, rows_per_simd);
    const auto num_tgs = at::ceil_div(total_simds, INNER_TG_SIZE / c10::metal::simdgroup_size);
    id<MTLComputeCommandEncoder> ce = stream->commandEncoder();
    auto ps = lib.getPipelineStateForFunc(kname);
    getMPSProfiler().beginProfileKernel(ps, prefix + "reduction_inner_chunk", {in}, stream);
    [ce setComputePipelineState:ps];
    const std::array<uint32_t, 4> sizes_s{num_rows, row_len, lanes, segments};
    mtl_setArgs(ce, in, out, sizes_s);
    if (divisor.has_value()) {
      mtl_setArgs<3>(ce, *divisor);
    }
    [ce dispatchThreads:MTLSizeMake(num_tgs * INNER_TG_SIZE, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(INNER_TG_SIZE, 1, 1)];
    getMPSProfiler().endProfileKernel(ps, stream);
  };
  // Segments per row for split-K: aim for ~SPLIT_TARGET_PARTIALS partials
  // (rows * segments) so pass 1 fills the GPU, cap so a segment keeps
  // >= SPLIT_MIN_SEG_LEN elements, then round so the segments come out
  // equal-sized.
  auto pick_split_segments = [](uint32_t num_rows, uint32_t row_len) -> uint32_t {
    const auto max_segs = std::min(SPLIT_MAX_SEGS, at::ceil_div(row_len, SPLIT_MIN_SEG_LEN));
    const auto segs =
        std::clamp(at::ceil_div(SPLIT_TARGET_PARTIALS, std::max(num_rows, 1u)), 2u, std::max(max_segs, 2u));
    return at::ceil_div(row_len, at::ceil_div(row_len, segs));
  };
  const auto nd = input_orig.dim();
  auto num_reduced = 0;
  auto reduced_dim = -1;
  for (auto d = 0; d < nd; d++) {
    if (input_orig.size(d) != output.size(d)) {
      num_reduced++;
      reduced_dim = d;
    }
  }
  // Two-pass paths divide on the final pass only, while the accumulator is
  // still in opmath_t; sum-family kernels always take the divisor buffer, so
  // pass 1 binds a no-op 0 (value ops take none).
  const std::optional<float> p1_div = opts.divisor.has_value() ? std::optional<float>(0.0f) : std::nullopt;

  // Narrow routing, shared by the contiguous and strided branches below:
  // batched or single-threadgroup single dispatch, or split-K two-pass. A
  // single narrow threadgroup saturates at ~NARROW_SPLIT_ELEMS_PER_TG
  // elements; past that, split the reduced dim into segments reduced in
  // parallel and fold the [num_segs, inner_size] partials in a second pass.
  auto route_narrow =
      [&](uint32_t dim_size, uint32_t inner_size, uint32_t outer_size, std::optional<std::array<uint32_t, 4>> strides) {
        const auto num_segs = outer_size > 1
            ? 1u
            : std::clamp<uint32_t>(
                  (static_cast<int64_t>(dim_size) * inner_size) / NARROW_SPLIT_ELEMS_PER_TG, 1u, SPLIT_MAX_SEGS);
        if (num_segs <= 1) {
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            @autoreleasepool {
              encode_narrow(input_orig,
                            output,
                            dim_size,
                            inner_size,
                            1,
                            outer_size,
                            opts.prefix,
                            opts.input_kernel_dtype,
                            opts.output_kernel_dtype,
                            opts.divisor,
                            strides);
            }
          });
          return;
        }
        auto partials = at::empty({(int64_t)num_segs, (int64_t)inner_size}, output.options().dtype(opts.partial_dtype));
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          @autoreleasepool {
            encode_narrow(input_orig,
                          partials,
                          dim_size,
                          inner_size,
                          num_segs,
                          1,
                          opts.prefix,
                          opts.input_kernel_dtype,
                          opts.partial_dtype,
                          p1_div,
                          strides);
            encode_narrow(partials,
                          output,
                          num_segs,
                          inner_size,
                          1,
                          1,
                          opts.pass2_prefix,
                          opts.partial_dtype,
                          opts.output_kernel_dtype,
                          opts.divisor,
                          std::nullopt);
          }
        });
      };

  // Non-innermost reduction on a non-contiguous input whose dims still
  // collapse to [outer_size, dim_size, inner_size] (e.g. a sliced or padded
  // view): the strided outer / outer_small_dim / narrow kernels index
  // through explicit strides, skipping the .contiguous() copy the generic
  // path would need. All kernels index in 32 bits.
  if (output.numel() > 1 && !input_orig.is_contiguous() && output.is_contiguous() && canUse32BitIndexMath(input_orig)) {
    if (num_reduced == 1 && reduced_dim < nd - 1) {
      c10::DimVector sizes(input_orig.sizes().begin(), input_orig.sizes().end());
      c10::DimVector strides(input_orig.strides().begin(), input_orig.strides().end());
      const auto [collapsed_dim, collapsed_ndim] = at::collapse_dims(sizes.data(), strides.data(), nd, reduced_dim);
      // Usable when at most one collapsed block remains on each side of the
      // reduced dim: [dim], [dim, inner], [outer, dim] or [outer, dim, inner].
      if (collapsed_ndim <= 2 || (collapsed_ndim == 3 && collapsed_dim == 1)) {
        const auto has_outer = collapsed_dim > 0;
        const auto has_inner = collapsed_dim < collapsed_ndim - 1;
        const auto outer_size = has_outer ? safe_downcast<uint32_t, int64_t>(sizes[0]) : 1u;
        const auto dim_size = safe_downcast<uint32_t, int64_t>(sizes[collapsed_dim]);
        const auto inner_size = has_inner ? safe_downcast<uint32_t, int64_t>(sizes[collapsed_dim + 1]) : 1u;
        const auto dim_stride = safe_downcast<uint32_t, int64_t>(strides[collapsed_dim]);
        const auto inner_stride = has_inner ? safe_downcast<uint32_t, int64_t>(strides[collapsed_dim + 1]) : 0u;
        const auto outer_stride = has_outer ? safe_downcast<uint32_t, int64_t>(strides[0]) : 0u;
        const auto natural_tgs = outer_size * c10::metal::ceil_div(inner_size, OUTER_TG_WIDTH);
        const auto use_small_dim = dim_size <= OUTER_SMALL_DIM_MAX_SIZE && natural_tgs >= SPLIT_MIN_TGS;
        // Only the sum family has narrow_strided kernels; value ops fall
        // through to the strided outer / small-dim layout below. Tensors
        // below CHUNK_MIN_NUMEL are enqueue-bound and skip the narrow path
        // (and its possible second dispatch) too. When the small-dim layout
        // is available and the reduced dim is short, its serial row walk
        // beats narrow's mostly-idle threadgroups.
        if (opts.has_strided_pass1 && inner_size < OUTER_TG_WIDTH && input_orig.numel() >= CHUNK_MIN_NUMEL &&
            !(use_small_dim && dim_size < NARROW_BATCHED_MIN_DIM_SIZE)) {
          route_narrow(
              dim_size, inner_size, outer_size, std::array<uint32_t, 4>{dim_stride, inner_stride, outer_stride, 0});
          return;
        }
        if (outer_size == 1 && natural_tgs < OUTER_SPLIT_MIN_TGS && dim_size >= OUTER_SPLIT_MIN_DIM_SIZE) {
          const auto num_segs =
              std::clamp(OUTER_SPLIT_STRIDED_TARGET_TGS / natural_tgs, 2u, std::min(dim_size, SPLIT_MAX_SEGS));
          auto partials =
              at::empty({(int64_t)num_segs, (int64_t)inner_size}, output.options().dtype(opts.partial_dtype));
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            @autoreleasepool {
              encode_outer_strided(input_orig,
                                   partials,
                                   dim_size,
                                   inner_size,
                                   num_segs,
                                   1,
                                   false,
                                   opts.prefix,
                                   opts.input_kernel_dtype,
                                   opts.partial_dtype,
                                   p1_div,
                                   dim_stride,
                                   inner_stride,
                                   outer_stride);
              encode_outer(partials,
                           output,
                           num_segs,
                           inner_size,
                           1,
                           1,
                           opts.pass2_prefix,
                           opts.partial_dtype,
                           opts.output_kernel_dtype,
                           opts.divisor);
            }
          });
          return;
        }
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          @autoreleasepool {
            encode_outer_strided(input_orig,
                                 output,
                                 dim_size,
                                 inner_size,
                                 1,
                                 outer_size,
                                 use_small_dim,
                                 opts.prefix,
                                 opts.input_kernel_dtype,
                                 opts.output_kernel_dtype,
                                 opts.divisor,
                                 dim_stride,
                                 inner_stride,
                                 outer_stride);
          }
        });
        return;
      }
    }
  }

  if (output.numel() > 1 && input_orig.is_contiguous() && output.is_contiguous()) {
    // Any non-innermost dim routes here, viewed as [outer_size, dim_size,
    // inner_size] with the dim reduced (outer_size = 1 when reduced_dim ==
    // 0); the kernels index in 32 bits.
    const auto non_innermost = num_reduced == 1 && reduced_dim != nd - 1 && canUse32BitIndexMath(input_orig);
    if (non_innermost) {
      const auto outer_size =
          safe_downcast<uint32_t, int64_t>(c10::multiply_integers(input_orig.sizes().slice(0, reduced_dim)));
      const auto dim_size = safe_downcast<uint32_t, int64_t>(input_orig.size(reduced_dim));
      const auto inner_size =
          safe_downcast<uint32_t, int64_t>(input_orig.numel() / (static_cast<int64_t>(outer_size) * dim_size));
      const auto natural_tgs = outer_size * c10::metal::ceil_div(inner_size, OUTER_TG_WIDTH);
      const auto use_small_dim = dim_size <= OUTER_SMALL_DIM_MAX_SIZE && natural_tgs >= SPLIT_MIN_TGS;
      // inner_size narrower than a threadgroup row: the narrow layout is the
      // only one that keeps a full threadgroup busy. Tensors below
      // CHUNK_MIN_NUMEL are enqueue-bound and stay on the single-dispatch
      // outer kernel below. When the small-dim layout is available and the
      // reduced dim is short, its serial row walk beats narrow's mostly-idle
      // threadgroups.
      if (inner_size < OUTER_TG_WIDTH && input_orig.numel() >= CHUNK_MIN_NUMEL &&
          !(use_small_dim && dim_size < NARROW_BATCHED_MIN_DIM_SIZE)) {
        route_narrow(dim_size, inner_size, outer_size, std::nullopt);
        return;
      }
      // Short reduced dim: a 32-row threadgroup would idle most rows, so use
      // the small-dim layout (one thread walks the whole reduced dim) when
      // there are enough columns to fill the GPU with 32-wide threadgroups.
      if (use_small_dim) {
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          @autoreleasepool {
            encode_outer_small_dim(input_orig,
                                   output,
                                   dim_size,
                                   inner_size,
                                   outer_size,
                                   opts.prefix,
                                   opts.input_kernel_dtype,
                                   opts.output_kernel_dtype,
                                   opts.divisor);
          }
        });
        return;
      }
      // Tall skinny case (few columns, long reduced dim): too few
      // threadgroups to fill the GPU, so split the reduced dim into segments
      // and fold the [num_segs, inner_size] partials in a second pass.
      if (outer_size == 1 && dim_size >= OUTER_SPLIT_MIN_DIM_SIZE && natural_tgs < OUTER_SPLIT_MIN_TGS) {
        const auto num_segs =
            std::min(dim_size / OUTER_SPLIT_MIN_SEG_LEN, std::max(2u, OUTER_SPLIT_MAX_TGS / natural_tgs));
        auto partials = at::empty({(int64_t)num_segs, (int64_t)inner_size}, output.options().dtype(opts.partial_dtype));
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          @autoreleasepool {
            encode_outer(input_orig,
                         partials,
                         dim_size,
                         inner_size,
                         num_segs,
                         1,
                         opts.prefix,
                         opts.input_kernel_dtype,
                         opts.partial_dtype,
                         p1_div);
            encode_outer(partials,
                         output,
                         num_segs,
                         inner_size,
                         1,
                         1,
                         opts.pass2_prefix,
                         opts.partial_dtype,
                         opts.output_kernel_dtype,
                         opts.divisor);
          }
        });
        return;
      }
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          encode_outer(input_orig,
                       output,
                       dim_size,
                       inner_size,
                       1,
                       outer_size,
                       opts.prefix,
                       opts.input_kernel_dtype,
                       opts.output_kernel_dtype,
                       opts.divisor);
        }
      });
      return;
    }
    // The inner kernels index in 32 bits.
    if (num_reduced == 1 && reduced_dim == input_orig.dim() - 1 && canUse32BitIndexMath(input_orig)) {
      const auto row_len = safe_downcast<uint32_t, int64_t>(input_orig.size(-1));
      const auto num_rows = safe_downcast<uint32_t, int64_t>(input_orig.numel() / row_len);
      // Tensors too small to fill the GPU gain nothing from packing rows
      // into simdgroups; keep them on the inner kernel below (the pre-chunk
      // routing, whose enqueue floor measures ~10% lower there).
      const bool tiny = input_orig.numel() < CHUNK_MIN_NUMEL;
      if (row_len <= CHUNK_MAX_ROW_LEN && !tiny) {
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          @autoreleasepool {
            encode_chunk(input_orig,
                         output,
                         num_rows,
                         row_len,
                         pick_lanes(row_len),
                         1u,
                         opts.prefix,
                         opts.input_kernel_dtype,
                         opts.output_kernel_dtype,
                         opts.divisor);
          }
        });
        return;
      }
      // Skinny-M/huge-K: one simdgroup per row would leave the GPU
      // under-occupied below SPLIT_MIN_TGS threadgroups, so split each
      // row into segments and fold the [num_rows, num_segs] partials in
      // pass 2.
      const auto inner_tgs = at::ceil_div(num_rows, INNER_TG_SIZE / c10::metal::simdgroup_size);
      if (inner_tgs < SPLIT_MIN_TGS && row_len >= SPLIT_MIN_ROW_LEN) {
        const auto num_segs = pick_split_segments(num_rows, row_len);
        auto partials = at::empty({(int64_t)num_rows, (int64_t)num_segs}, output.options().dtype(opts.partial_dtype));
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          @autoreleasepool {
            encode_chunk(input_orig,
                         partials,
                         num_rows,
                         row_len,
                         c10::metal::simdgroup_size,
                         num_segs,
                         opts.prefix,
                         opts.input_kernel_dtype,
                         opts.partial_dtype,
                         p1_div);
            encode_inner(partials,
                         output,
                         num_rows,
                         num_segs,
                         opts.pass2_prefix,
                         opts.partial_dtype,
                         opts.output_kernel_dtype,
                         opts.divisor);
          }
        });
        return;
      }
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          encode_inner(input_orig,
                       output,
                       num_rows,
                       row_len,
                       opts.prefix,
                       opts.input_kernel_dtype,
                       opts.output_kernel_dtype,
                       opts.divisor);
        }
      });
      return;
    }
  }

  // Two-pass for large full reductions: pass 1 splits input into <=512
  // contiguous slices, each TG reduces one slice to a partial; pass 2
  // collapses the num_groups partials into the final scalar.
  if (output.numel() == 1 && reduction_size > MAX_THREADGROUP_SIZE * NCHAINS) {
    auto num_groups = std::min(512u, c10::metal::ceil_div(reduction_size, MAX_THREADGROUP_SIZE * NCHAINS));
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
      auto partials = at::empty({(int64_t)num_groups}, output.options().dtype(opts.partial_dtype));

      auto p1_kernel =
          fmt::format("{}reduction{}_{}_{}", opts.prefix, use_strided ? "_strided" : "_flat", in_str, partial_str);
      auto p2_kernel = fmt::format("{}reduction_{}_{}", opts.pass2_prefix, partial_str, out_str);

      NormParams params1{};
      params1.reduction_size = elems_per_group;
      if (use_strided) {
        params1.ndim = input.dim();
        for (const auto d : c10::irange(input.dim())) {
          params1.input_sizes[d] = input.size(d);
          params1.input_strides[d] = input.stride(d);
        }
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
          getMPSProfiler().beginProfileKernel(ps1, opts.prefix + "reduction_pass1", {input}, stream);
          [ce setComputePipelineState:ps1];
          if (use_strided) {
            mtl_setArgs(ce, input, partials, params1);
            // Round up to a full simdgroup: c10::metal::simd_max/min<long>
            // reads its neighbours' registers, and a partially populated
            // simdgroup yields undefined data (0 in practice) rather than the
            // op's identity. Padding threads skip the load loop (tid >= rsize)
            // and contribute Op::identity() instead.
            auto tpg1 = std::min(MAX_THREADGROUP_SIZE, c10::metal::round_up(elems_per_group, 32u));
            [ce dispatchThreads:MTLSizeMake(num_groups * tpg1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg1, 1, 1)];
          } else {
            const std::array<uint32_t, 2> fparams{num_groups, elems_per_group};
            mtl_setArgs(ce, input, partials, fparams);
            constexpr uint32_t TPG = 256;
            [ce dispatchThreads:MTLSizeMake(num_groups * TPG, 1, 1) threadsPerThreadgroup:MTLSizeMake(TPG, 1, 1)];
          }
          getMPSProfiler().endProfileKernel(ps1, stream);

          auto ps2 = lib.getPipelineStateForFunc(p2_kernel);
          getMPSProfiler().beginProfileKernel(ps2, opts.prefix + "reduction_pass2", {partials}, stream);
          [ce setComputePipelineState:ps2];
          mtl_setArgs(ce, partials, output, params2);
          auto tpg2 = std::min(MAX_THREADGROUP_SIZE, c10::metal::round_up(num_groups, 32u));
          [ce dispatchThreads:MTLSizeMake(tpg2, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg2, 1, 1)];
          getMPSProfiler().endProfileKernel(ps2, stream);
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
      getMPSProfiler().beginProfileKernel(ps, opts.prefix + "reduction", {input_orig}, stream);
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
      getMPSProfiler().endProfileKernel(ps, stream);
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
                             .partial_dtype = at::toOpMathType(output.scalar_type()),
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
