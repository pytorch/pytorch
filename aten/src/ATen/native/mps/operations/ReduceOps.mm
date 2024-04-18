//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cdist_forward_native.h>
#include <ATen/ops/all_native.h>
#include <ATen/ops/amax_native.h>
#include <ATen/ops/amin_native.h>
#include <ATen/ops/aminmax_native.h>
#include <ATen/ops/any_native.h>
#include <ATen/ops/argmax_native.h>
#include <ATen/ops/argmin_native.h>
#include <ATen/ops/count_nonzero_native.h>
#include <ATen/ops/linalg_vector_norm_native.h>
#include <ATen/ops/max_native.h>
#include <ATen/ops/mean_native.h>
#include <ATen/ops/median.h>
#include <ATen/ops/median_native.h>
#include <ATen/ops/min_native.h>
#include <ATen/ops/nansum_native.h>
#include <ATen/ops/norm_native.h>
#include <ATen/ops/prod_native.h>
#include <ATen/ops/std_mean_native.h>
#include <ATen/ops/std_native.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/sum_native.h>
#include <ATen/ops/trace_native.h>
#include <ATen/ops/var_mean_native.h>
#include <ATen/ops/var_native.h>
#endif

namespace at::native {
namespace mps {
typedef MPSGraphTensor* (^NormOpBlock)(mps::MPSBinaryCachedGraph*, MPSGraphTensor*, MPSGraphTensor*);
#define NormOpFn(graph, primary, secondary) \
  MPSGraphTensor*(mps::MPSBinaryCachedGraph * graph, MPSGraphTensor * primary, MPSGraphTensor * secondary)

enum StdVarType { STANDARD_VARIANCE, STANDARD_DEVIATION };

enum MPSReductionType {
  MAX,
  MIN,
  AMAX,
  AMIN,
  SUM,
  PROD,
  MEAN,
  COUNT_NONZERO,
  TRACE,
  NANSUM,
};

using namespace mps;

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
                              c10::optional<ScalarType> dtype,
                              const Tensor& output_t,
                              MPSReductionType reduction_type,
                              const std::string& func_name) {
  bool macOS13_3_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
  MPS_CHECK_INT64_OP_SUPPORTED(input_t, macOS13_3_plus, func_name);
  bool canSqueezeLastDim = true;
  IntArrayRef input_shape = input_t.sizes();
  if (opt_dim.has_value()) {
    IntArrayRef dim = opt_dim.value();
    for (const auto dim_val : dim) {
      auto wrap_dim = maybe_wrap_dim(dim_val, input_shape.size());
      if (wrap_dim >= 4) {
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
  NSArray<NSNumber*>* wrappedAxes = mps::getTensorAxes(input_shape, opt_dim);

  if (output_t.numel() == 0 || input_t.numel() == 0) {
    if (reduction_type == MPSReductionType::PROD) {
      output_t.fill_(1);
    } else if (reduction_type == MPSReductionType::SUM) {
      output_t.zero_();
    }
    return;
  }
  auto stream = at::mps::getCurrentMPSStream();
  @autoreleasepool {
    std::string dtype_str = dtype.has_value() ? mps::getMPSTypeString(dtype.value()) : "";
    NSString* ns_key = [[wrappedAxes valueForKey:@"description"] componentsJoinedByString:@","];
    string key = func_name + ":" + string([ns_key UTF8String]) + ":" + getTensorsStringKey(input_t) + ":" +
        std::to_string(keepdim) + ":" + std::to_string(reduction_type) + ":" + getTensorsStringKey(output_t) + ":" +
        dtype_str;
    using CachedGraph = MPSUnaryCachedGraph;
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputScalarType = input_t.scalar_type();

      MPSGraphTensor* inputTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input_t.scalar_type()), mpsShape);
      MPSGraphTensor* castInputTensor = inputTensor;
      MPSDataType inputCastType = MPSDataTypeInvalid;
      if (dtype.has_value() &&
          (dtype.value() == kFloat || dtype.value() == kHalf || dtype.value() == kInt ||
           (dtype.value() == kLong && macOS13_3_plus))) {
        inputCastType = getMPSDataType(dtype.value());
      } else if (inputScalarType != kInt && inputScalarType != kHalf && inputScalarType != kFloat &&
                 inputScalarType != kComplexFloat && inputScalarType != kComplexHalf &&
                 (inputScalarType != kLong || !macOS13_3_plus)) {
        inputCastType = getMPSDataType(kFloat);
      } else if (!is_macos_13_or_newer() && inputScalarType == kHalf) {
        inputCastType = getMPSDataType(kFloat);
      }

      if (inputCastType != MPSDataTypeInvalid) {
        castInputTensor = castMPSTensor(mpsGraph, inputTensor, inputCastType);
      }

      MPSGraphTensor* castOutputTensor = nil;

      if (reduction_type == MPSReductionType::SUM) {
        castOutputTensor = [mpsGraph reductionSumWithTensor:castInputTensor axes:wrappedAxes name:nil];
      } else if (reduction_type == MPSReductionType::PROD) {
        castOutputTensor = [mpsGraph reductionProductWithTensor:castInputTensor axes:wrappedAxes name:nil];
      } else if (reduction_type == MPSReductionType::MEAN) {
        castOutputTensor = [mpsGraph meanOfTensor:castInputTensor axes:wrappedAxes name:nil];
      } else if (reduction_type == MPSReductionType::COUNT_NONZERO) {
        MPSGraphTensor* zeros = [mpsGraph constantWithScalar:0 dataType:castInputTensor.dataType];

        MPSGraphTensor* nonZeros = [mpsGraph notEqualWithPrimaryTensor:castInputTensor secondaryTensor:zeros name:nil];

        castOutputTensor = [mpsGraph reductionSumWithTensor:nonZeros axes:wrappedAxes name:nil];
      } else if (reduction_type == MPSReductionType::AMAX) {
        castOutputTensor = [mpsGraph reductionMaximumWithTensor:castInputTensor axes:wrappedAxes name:nil];
      } else if (reduction_type == MPSReductionType::AMIN) {
        castOutputTensor = [mpsGraph reductionMinimumWithTensor:castInputTensor axes:wrappedAxes name:nil];
      } else if (reduction_type == MPSReductionType::TRACE) {
        MPSGraphTensor* bandPartWithTensor = [mpsGraph bandPartWithTensor:castInputTensor
                                                                 numLower:0
                                                                 numUpper:0
                                                                     name:nil];
        castOutputTensor = [mpsGraph reductionSumWithTensor:bandPartWithTensor axes:@[ @0, @1 ] name:nil];
      } else if (reduction_type == MPSReductionType::NANSUM) {
        // Create a 0 tensor of the same shape as inputTensor
        MPSGraphTensor* zeros = [mpsGraph constantWithScalar:0.0 dataType:castInputTensor.dataType];
        // Find NaNs
        MPSGraphTensor* nanMask = [mpsGraph isNaNWithTensor:castInputTensor name:nil];
        // Replace NaNs with 0
        MPSGraphTensor* nanReplaced = [mpsGraph selectWithPredicateTensor:nanMask
                                                      truePredicateTensor:zeros
                                                     falsePredicateTensor:castInputTensor
                                                                     name:nil];
        // Sum
        castOutputTensor = [mpsGraph reductionSumWithTensor:nanReplaced axes:wrappedAxes name:nil];
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

static void impl_func_norm_mps(const Tensor& input_tensor,
                               const Tensor& other_tensor,
                               const OptionalScalarRef& opt_p,
                               IntArrayRef dim,
                               bool keepdim,
                               c10::optional<ScalarType> opt_dtype,
                               const Tensor& output_t,
                               bool cdist = false,
                               c10::optional<IntArrayRef> input_broadcasted_shape = c10::nullopt,
                               NormOpBlock normOpBlock = nullptr) {
  auto p = opt_p.has_value() ? opt_p.get().to<double>() : Scalar(2.0).to<double>();
  if (input_tensor.numel() == 0) {
    output_t.fill_((p < 0) ? INFINITY : 0);
    return;
  }

  auto input_t = (input_tensor.sizes().size() == 0) ? input_tensor.view({1}) : input_tensor;
  auto in_dtype = opt_dtype.value_or(input_tensor.scalar_type());
  auto mps_input_dtype = getMPSDataType(in_dtype);
  TORCH_CHECK(!input_tensor.is_complex(), "norm ops are not supported for complex yet");

  IntArrayRef input_shape = cdist ? input_broadcasted_shape.value() : input_t.sizes();

  for (const auto dim_val : dim) {
    auto wrap_dim = maybe_wrap_dim(dim_val, input_shape.size());
    TORCH_CHECK(wrap_dim < static_cast<decltype(wrap_dim)>(input_shape.size()),
                "norm_out_mps: reduction dim must be in the range of input shape")
  }

  auto reciprocal_p = 1 / p;
  bool pIsZero = (p == 0.0);
  bool pIsPosInf = (p == std::numeric_limits<double>::infinity());
  bool pIsNegInf = (p == -std::numeric_limits<double>::infinity());

  int64_t num_input_dims = input_shape.size();
  int64_t num_reduce_dims = dim.size();
  int64_t num_output_dims;

  // For output shape calculation, assume that keepdim is true
  num_output_dims = num_input_dims;
  NSMutableArray<NSNumber*>* apparent_output_shape = nil;
  NSMutableArray<NSNumber*>* apparent_input_shape = nil;

  // Reduction axes
  NSMutableArray<NSNumber*>* axes;
  set_axes(axes, num_reduce_dims, dim, input_shape.size());

  set_apparent_shapes(apparent_output_shape, apparent_input_shape, num_reduce_dims, num_output_dims, input_shape, axes);

  NSArray<NSNumber*>* wrappedAxes = mps::getTensorAxes(input_shape, dim);
  if (cdist) {
    apparent_input_shape = [mps::getMPSShape(input_tensor.sizes()) mutableCopy];
    apparent_output_shape = [mps::getMPSShape(output_t.sizes()) mutableCopy];
  }

  if (output_t.numel() == 0) {
    return;
  }

  // Cast FP16 to FP32 on macOS Monterey due to precision issues.
  // This is fixed starting with macOS Ventura.
  bool castInputData = false;
  if (!is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_0_PLUS) && mps_input_dtype == MPSDataTypeFloat16) {
    castInputData = true;
    mps_input_dtype = MPSDataTypeFloat32;
  }

  auto stream = at::mps::getCurrentMPSStream();
  @autoreleasepool {
    NSString* ns_key = [[wrappedAxes valueForKey:@"description"] componentsJoinedByString:@","];
    string keepdim_info = (keepdim) ? "keepdim=1" : "keepdim=0";
    string tensor_key = cdist ? getTensorsStringKey({input_tensor, other_tensor}) : getTensorsStringKey({input_t});
    string key = string("norm_out_mps:") + [ns_key UTF8String] + ":" + tensor_key + ":p" + to_string(p) + ":" +
        keepdim_info + ":" + toString(in_dtype) + ":" + to_string(castInputData);

    auto cachedGraph = LookUpOrCreateCachedGraph<MPSBinaryCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->inputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, input_tensor);

      if (cdist) {
        newCachedGraph->otherTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, other_tensor);
      }

      MPSGraphTensor* inputTensor = cdist
          ? normOpBlock(newCachedGraph, newCachedGraph->inputTensor_, newCachedGraph->otherTensor_)
          : newCachedGraph->inputTensor_;

      if (opt_dtype.has_value() || castInputData) {
        inputTensor = castMPSTensor(mpsGraph, inputTensor, mps_input_dtype);
      }

      MPSGraphTensor* outputTensor;

      if (pIsZero) {
        MPSGraphTensor* zeros = [mpsGraph constantWithScalar:0.0 dataType:mps_input_dtype];
        MPSGraphTensor* ones = [mpsGraph constantWithScalar:1.0 dataType:mps_input_dtype];
        MPSGraphTensor* nonZeros = [mpsGraph selectWithPredicateTensor:inputTensor
                                                   truePredicateTensor:ones
                                                  falsePredicateTensor:zeros
                                                                  name:nil];
        outputTensor = [mpsGraph reductionSumWithTensor:nonZeros axes:wrappedAxes name:nil];
      } else if (pIsPosInf) {
        MPSGraphTensor* absoluteTensor = [mpsGraph absoluteWithTensor:inputTensor name:nil];
        outputTensor = [mpsGraph reductionMaximumWithTensor:absoluteTensor axes:wrappedAxes name:nil];
      } else if (pIsNegInf) {
        MPSGraphTensor* absoluteTensor = [mpsGraph absoluteWithTensor:inputTensor name:nil];
        outputTensor = [mpsGraph reductionMinimumWithTensor:absoluteTensor axes:wrappedAxes name:nil];
      } else {
        MPSGraphTensor* absoluteTensor = [mpsGraph absoluteWithTensor:inputTensor name:nil];

        MPSGraphTensor* powerValTensor = [mpsGraph constantWithScalar:p dataType:mps_input_dtype];

        MPSGraphTensor* reciprocalPowerValTensor = [mpsGraph constantWithScalar:reciprocal_p dataType:mps_input_dtype];

        MPSGraphTensor* powerTensor = [mpsGraph powerWithPrimaryTensor:absoluteTensor
                                                       secondaryTensor:powerValTensor
                                                                  name:nil];

        MPSGraphTensor* reductionSumTensor = [mpsGraph reductionSumWithTensor:powerTensor axes:wrappedAxes name:nil];

        outputTensor = [mpsGraph powerWithPrimaryTensor:reductionSumTensor
                                        secondaryTensor:reciprocalPowerValTensor
                                                   name:nil];
      }

      if (cdist) {
        outputTensor = [mpsGraph reshapeTensor:outputTensor withShape:mps::getMPSShape(output_t) name:nil];
      }

      newCachedGraph->outputTensor_ =
          castInputData ? castMPSTensor(mpsGraph, outputTensor, output_t.scalar_type()) : outputTensor;
    });

    auto otherPlaceholder = Placeholder();
    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t, apparent_output_shape);

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary dictionary];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();

    if (cdist) {
      otherPlaceholder = Placeholder(cachedGraph->otherTensor_, other_tensor);
      feeds[otherPlaceholder.getMPSGraphTensor()] = otherPlaceholder.getMPSGraphTensorData();
    }

    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

static Tensor std_var_common_impl_mps(const Tensor& input_t,
                                      at::OptionalIntArrayRef dim,
                                      const c10::optional<Scalar>& correction,
                                      bool keepdim,
                                      StdVarType stdVarType) {
  using CachedGraph = MPSUnaryCachedGraph;

  IntArrayRef input_shape = input_t.sizes();
  int64_t num_input_dims = input_shape.size();

  bool use_dim = dim.has_value();
  IntArrayRef dim_value = use_dim ? dim.value() : NULL;

  if (use_dim) {
    string errMessage = (stdVarType == STANDARD_DEVIATION) ? "std_mps" : "var_mps";
    errMessage += ": reduction dim must be in the range of input shape";
    for (const auto dim : dim_value) {
      auto wrap_dim = maybe_wrap_dim(dim, num_input_dims);
      TORCH_CHECK(wrap_dim < static_cast<decltype(wrap_dim)>(input_shape.size()), errMessage.c_str())
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
  } else if (!keepdim && use_dim && dim_value.size() > 0) {
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
  } else if ((keepdim && !use_dim) || (keepdim && use_dim && dim_value.size() <= 0)) {
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
  } else if (keepdim && use_dim && dim_value.size() > 0) {
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
                              c10::nullopt,
                              kMPS,
                              c10::nullopt,
                              c10::nullopt);

  if (output_t.numel() == 0 || input_t.numel() == 0) {
    return output_t;
  }

  double dof = std::max(0.0, correction_n - correction_value);
  double bessel_correction = correction_n / dof;
  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    string op_key = (stdVarType == STANDARD_DEVIATION) ? "std_mps" : "var_mps";
    NSString* ns_key = [[wrappedAxes valueForKey:@"description"] componentsJoinedByString:@","];
    string bessel_corrected = (use_correction && correction_value) ? "unbiased " : "biased ";
    string use_dim_info = (use_dim) ? "use_dim=1:" + to_string(dim_value.size()) : "use_dim=0";
    string keepdim_info = (keepdim) ? "keepdim=1" : "keepdim=0";
    string key = op_key + ":" + getTensorsStringKey(input_t) + ":" + use_dim_info + ":" + keepdim_info + ":" +
        string([ns_key UTF8String]) + ":" + bessel_corrected + ":" + std::to_string(correction_value);

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

static Tensor min_max_mps_impl(const Tensor& input_t, MPSReductionType reduction_type, const std::string& func_name) {
  bool macOS13_3_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
  MPS_CHECK_INT64_OP_SUPPORTED(input_t, macOS13_3_plus, "min_max");

  using CachedGraph = MPSUnaryCachedGraph;

  IntArrayRef input_shape = input_t.sizes();
  int64_t num_in_elements = c10::multiply_integers(input_shape);

  Tensor output_t = at::empty({}, input_t.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);

  if (output_t.numel() == 0 || num_in_elements == 0) {
    return output_t;
  }

  @autoreleasepool {
    string key = func_name + mps::getTensorsStringKey(input_t);
    CachedGraph* cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_t);

      MPSGraphTensor* castOutputTensor = nil;
      MPSGraphTensor* castInputTensor =
          castToIHFTypes(mpsGraph, inputTensor, input_t, /*includesInt64=*/macOS13_3_plus);

      NSArray<NSNumber*>* axes = getTensorAxes(input_t);
      if (reduction_type == MPSReductionType::MAX) {
        castOutputTensor = [mpsGraph reductionMaximumWithTensor:castInputTensor axes:axes name:nil];
      } else if (reduction_type == MPSReductionType::MIN) {
        castOutputTensor = [mpsGraph reductionMinimumWithTensor:castInputTensor axes:axes name:nil];
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
  bool macOS13_3_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
  MPS_CHECK_INT64_OP_SUPPORTED(input_t, macOS13_3_plus, "min_max_out");

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

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    string key = func_name + getTensorsStringKey({input_t, indices_t}) + ":" + to_string(dim_);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_t);
      MPSGraphTensor* outputTensor = nil;
      MPSGraphTensor* castInputTensor =
          castToIHFTypes(mpsGraph, inputTensor, input_t, /*includesInt64=*/macOS13_3_plus);

      if (reduction_type == MPSReductionType::MAX) {
        outputTensor = [mpsGraph reductionMaximumWithTensor:castInputTensor axis:(NSInteger)dim_ name:nil];
      } else if (reduction_type == MPSReductionType::MIN) {
        outputTensor = [mpsGraph reductionMinimumWithTensor:castInputTensor axis:(NSInteger)dim_ name:nil];
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
        at::empty(IntArrayRef(vec_out_shape), input_t.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
    indices_t = at::empty(IntArrayRef(vec_out_shape), ScalarType::Long, c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
  } else {
    output_t = at::empty(
        IntArrayRef(vec_apparent_out_shape), input_t.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
    indices_t = at::empty(
        IntArrayRef(vec_apparent_out_shape), ScalarType::Long, c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
  }

  if (output_t.numel() == 0 || input_t.numel() == 0) {
    return std::tuple<Tensor, Tensor>{output_t, indices_t};
  }

  min_max_out_mps(input_t, dim, keepdim, output_t, indices_t, reduction_type, func_name);

  return std::tuple<Tensor, Tensor>{output_t, indices_t};
}

static void argmax_argmin_out_mps(const Tensor& input_t,
                                  c10::optional<int64_t> dim,
                                  bool keepdim,
                                  const Tensor& output_t,
                                  MPSReductionType reduction_type,
                                  const std::string& func_name) {
  using CachedGraph = MPSUnaryCachedGraph;

  bool macOS13_3_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
  MPS_CHECK_INT64_OP_SUPPORTED(input_t, macOS13_3_plus, "argmax_argmin_out");

  int64_t dim_ = -1;

  if (dim.has_value()) {
    dim_ = maybe_wrap_dim(dim.value(), input_t.dim());
    zero_numel_check_dims(input_t, dim_, reduction_type == MPSReductionType::MAX ? "argmax()" : "argmin()");
  } else {
    TORCH_CHECK_INDEX(input_t.numel() != 0,
                      reduction_type == MPSReductionType::MAX ? "argmax()" : "argmin()",
                      ": Expected reduction dim to be specified for input.numel() == 0.");
    // Since input will be flattened, take argmax or argmin along 0'th dimension
    dim_ = 0;
  }

  // Calculate the output shape according to keepdim=True
  // If there is no dim argument, the input shape is flattened
  IntArrayRef input_shape = input_t.sizes();
  int64_t num_input_dims = input_shape.size();
  NSMutableArray<NSNumber*>* apparent_in_shape = nil;
  NSMutableArray<NSNumber*>* apparent_out_shape = nil;

  if (dim.has_value()) {
    apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
    for (const auto i : c10::irange(num_input_dims)) {
      apparent_out_shape[i] = dim_ == i ? @1 : [NSNumber numberWithInt:input_shape[i]];
    }
  } else {
    apparent_in_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    int64_t num_in_elements = c10::multiply_integers(input_shape);
    apparent_in_shape[0] = [NSNumber numberWithInt:num_in_elements];

    apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
    apparent_out_shape[0] = @1;
  }

  if (output_t.numel() == 0) {
    return;
  }

  if (!apparent_in_shape) {
    apparent_in_shape = [getMPSShape(input_t.sizes()) mutableCopy];
  }

  auto stream = at::mps::getCurrentMPSStream();
  @autoreleasepool {
    NSString* ns_key = [[apparent_in_shape valueForKey:@"description"] componentsJoinedByString:@","];
    string key =
        func_name + ":" + to_string(dim_) + ":" + getTensorsStringKey(input_t) + ":" + string([ns_key UTF8String]);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputScalarType = input_t.scalar_type();
      MPSGraphTensor* inputTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(inputScalarType), apparent_in_shape);
      MPSGraphTensor* argreduceOutTensor = nil;

      MPSGraphTensor* castInputTensor = inputTensor;
      if (inputScalarType != kInt && inputScalarType != kHalf && inputScalarType != kFloat &&
          (inputScalarType != kLong || !macOS13_3_plus)) {
        castInputTensor = castMPSTensor(mpsGraph, inputTensor, kFloat);
      }
      if (reduction_type == MPSReductionType::MAX) {
        argreduceOutTensor = [mpsGraph reductionArgMaximumWithTensor:castInputTensor axis:(NSInteger)dim_ name:nil];
      } else {
        argreduceOutTensor = [mpsGraph reductionArgMinimumWithTensor:castInputTensor axis:(NSInteger)dim_ name:nil];
      }

      MPSGraphTensor* outputTensor = argreduceOutTensor;
      if (getMPSDataType(output_t) != [argreduceOutTensor dataType]) {
        outputTensor = castMPSTensor(mpsGraph, argreduceOutTensor, output_t.scalar_type());
      }

      MPSGraphTensor* outputClampedTensor =
          [mpsGraph clampWithTensor:outputTensor
                     minValueTensor:[mpsGraph constantWithScalar:0 dataType:MPSDataTypeInt64]
                     maxValueTensor:[mpsGraph constantWithScalar:0x7FEFFFFFFFFFFFFF dataType:MPSDataTypeInt64]
                               name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputClampedTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t, apparent_in_shape);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t, apparent_out_shape);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

} // namespace mps

TORCH_IMPL_FUNC(sum_out_mps)
(const Tensor& input_t,
 OptionalIntArrayRef opt_dim,
 bool keepdim,
 c10::optional<ScalarType> dtype,
 const Tensor& output_t) {
  mps::reduction_out_mps(input_t, opt_dim, keepdim, dtype, output_t, mps::MPSReductionType::SUM, "sum_out_mps");
}

Tensor& nansum_out_mps(const Tensor& self,
                       OptionalIntArrayRef dim,
                       bool keepdim,
                       c10::optional<ScalarType> opt_dtype,
                       Tensor& result) {
  TORCH_CHECK(!c10::isComplexType(self.scalar_type()), "nansum does not support complex inputs");
  if (c10::isIntegralType(self.scalar_type(), true)) {
    return at::sum_out(result, self, dim, keepdim, opt_dtype);
  }
  ScalarType dtype = get_dtype_from_result(result, opt_dtype);
  const auto mask = make_dim_mask(dim, self.dim());
  resize_reduction_result(result, self, mask, keepdim, dtype);
  mps::reduction_out_mps(self, dim, keepdim, dtype, result, mps::MPSReductionType::NANSUM, "nansum_out_mps");
  return result;
}

Tensor nansum_mps(const Tensor& self, OptionalIntArrayRef dim, bool keepdim, c10::optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
  Tensor result = create_reduction_result(self, dim, keepdim, dtype);
  return nansum_out_mps(self, dim, keepdim, dtype, result);
}

Tensor trace_mps(const Tensor& self) {
  TORCH_CHECK(self.dim() == 2, "trace: expected a matrix, but got tensor with dim ", self.dim());

  Tensor output_t =
      at::empty({}, get_dtype_from_self(self, c10::nullopt, true), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);

  std::vector<int64_t> dims(self.dim());
  std::iota(dims.begin(), dims.end(), 0);

  mps::reduction_out_mps(self,
                         IntArrayRef(dims),
                         false,
                         c10::nullopt,
                         const_cast<Tensor&>(output_t),
                         mps::MPSReductionType::TRACE,
                         "trace_mps");

  return output_t;
}

TORCH_IMPL_FUNC(prod_out_mps)
(const Tensor& input_t, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype, const Tensor& output_t) {
  int64_t dims[1] = {dim};
  mps::reduction_out_mps(
      input_t, IntArrayRef(dims, 1), keepdim, dtype, output_t, mps::MPSReductionType::PROD, "prod_out_mps");
}

TORCH_IMPL_FUNC(amax_out_mps)(const Tensor& input_t, IntArrayRef dim, bool keepdim, const Tensor& output_t) {
  mps::reduction_out_mps(input_t, dim, keepdim, c10::nullopt, output_t, mps::MPSReductionType::AMAX, "amax_out_mps");
}

TORCH_IMPL_FUNC(amin_out_mps)(const Tensor& input_t, IntArrayRef dim, bool keepdim, const Tensor& output_t) {
  mps::reduction_out_mps(input_t, dim, keepdim, c10::nullopt, output_t, mps::MPSReductionType::AMIN, "amin_out_mps");
}

TORCH_IMPL_FUNC(aminmax_out_mps)
(const Tensor& input_t, c10::optional<int64_t> dim_opt, bool keepdim, const Tensor& min_t, const Tensor& max_t) {
  mps::reduction_out_mps(input_t,
                         dim_opt.has_value() ? OptionalIntArrayRef({*dim_opt}) : c10::nullopt,
                         keepdim,
                         c10::nullopt,
                         min_t,
                         mps::MPSReductionType::AMIN,
                         "aminmax_out_mps_min");
  mps::reduction_out_mps(input_t,
                         dim_opt.has_value() ? OptionalIntArrayRef({*dim_opt}) : c10::nullopt,
                         keepdim,
                         c10::nullopt,
                         max_t,
                         mps::MPSReductionType::AMAX,
                         "aminmax_out_mps_max");
}

Tensor prod_mps(const Tensor& self, c10::optional<ScalarType> opt_dtype) {
  std::vector<int64_t> dims(self.dim());
  std::iota(dims.begin(), dims.end(), 0);

  Tensor output_t =
      at::empty({}, get_dtype_from_self(self, opt_dtype, true), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);

  mps::reduction_out_mps(self,
                         IntArrayRef(dims),
                         false,
                         opt_dtype,
                         const_cast<Tensor&>(output_t),
                         mps::MPSReductionType::PROD,
                         "prod_mps");

  return output_t;
}

Tensor count_nonzero_mps(const Tensor& self, IntArrayRef dims) {
  int64_t shape_size = dims.size() == 0 ? 0 : self.sizes().size() - dims.size();
  int64_t out_shape = std::max(shape_size, 0LL);
  std::vector<int64_t> output_shape(out_shape);
  std::vector<int64_t> dims_vec = dims.vec();
  std::for_each(dims_vec.begin(), dims_vec.end(), [&](int64_t& n) { n = maybe_wrap_dim(n, self); });

  if (out_shape != 0) {
    int out_dim = 0;
    for (const auto self_dim : c10::irange((self.sizes().size()))) {
      if (std::find(dims_vec.begin(), dims_vec.end(), self_dim) == dims_vec.end()) {
        output_shape[out_dim++] = (self.sizes()[self_dim]);
      }
    }
  }

  Tensor output_t =
      at::empty(IntArrayRef(output_shape), ScalarType::Long, c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
  mps::reduction_out_mps(self,
                         dims,
                         false,
                         self.scalar_type(),
                         const_cast<Tensor&>(output_t),
                         mps::MPSReductionType::COUNT_NONZERO,
                         "count_nonzero_mps");

  return output_t;
}

TORCH_IMPL_FUNC(mean_out_mps)
(const Tensor& input_t,
 OptionalIntArrayRef opt_dim,
 bool keepdim,
 c10::optional<ScalarType> dtype,
 const Tensor& output_t) {
  mps::reduction_out_mps(input_t, opt_dim, keepdim, dtype, output_t, mps::MPSReductionType::MEAN, "mean_out_mps");
}

TORCH_IMPL_FUNC(norm_out_mps)
(const Tensor& self, const OptionalScalarRef opt_p, IntArrayRef dim, bool keepdim, const Tensor& result) {
  mps::impl_func_norm_mps(self, self, opt_p, dim, keepdim, c10::nullopt, result, /*cdist=*/false);
}

TORCH_IMPL_FUNC(norm_dtype_out_mps)
(const Tensor& self,
 const OptionalScalarRef opt_p,
 IntArrayRef dim,
 bool keepdim,
 ScalarType dtype,
 const Tensor& result) {
  mps::impl_func_norm_mps(self, self, opt_p, dim, keepdim, dtype, result, /*cdist=*/false);
}

TORCH_IMPL_FUNC(linalg_vector_norm_out_mps)
(const Tensor& self,
 const Scalar& scalar_ord,
 OptionalIntArrayRef opt_dim,
 bool keepdim,
 c10::optional<ScalarType> opt_dtype,
 const Tensor& result) {
  mps::impl_func_norm_mps(
      self, self, scalar_ord, opt_dim.value_or(IntArrayRef{}), keepdim, opt_dtype, result, /*cdist=*/false);
}

Tensor _cdist_forward_mps(const Tensor& x1, const Tensor& x2, const double p, c10::optional<int64_t> compute_mode) {
  using namespace mps;
  TORCH_CHECK(x1.dim() >= 2, "cdist only supports at least 2D tensors, X1 got: ", x1.dim(), "D");
  TORCH_CHECK(x2.dim() >= 2, "cdist only supports at least 2D tensors, X2 got: ", x2.dim(), "D");
  TORCH_CHECK(x1.size(-1) == x2.size(-1),
              "X1 and X2 must have the same number of columns. X1: ",
              x1.size(-1),
              " X2: ",
              x2.size(-1));
  TORCH_CHECK(
      at::isFloatingType(x1.scalar_type()), "cdist only supports floating-point dtypes, X1 got: ", x1.scalar_type());
  auto device1 = x1.device().type();
  TORCH_CHECK(
      at::isFloatingType(x2.scalar_type()), "cdist only supports floating-point dtypes, X2 got: ", x2.scalar_type());
  auto device2 = x2.device().type();
  TORCH_CHECK(p >= 0, "cdist only supports non-negative p values");
  TORCH_CHECK(device1 == device2, "X1 and X2 must have the same device type. X1: ", device1, " X2: ", device2);
  TORCH_CHECK(x1.is_mps() && (x1.get_device() == x2.get_device()),
              "device of X1 (",
              x1.get_device(),
              ") must match device of X2 (",
              x2.get_device(),
              ")");

  int64_t c1 = x1.size(-1);
  int64_t c2 = x2.size(-1);

  auto dim1 = x1.dim();
  auto dim2 = x2.dim();
  int64_t mode = compute_mode.value_or(0);
  TORCH_CHECK(mode >= 0 && mode <= 2, "possible modes: 0, 1, 2, but was: ", mode);

  int64_t r1 = x1.size(-2);
  int64_t r2 = x2.size(-2);

  // For batch calculation we expand all dimensions(except the last two) to one, with size that equals to product of
  // them. The last two dimensions will stay the same
  IntArrayRef batch_tensor1(x1.sizes().data(), dim1 - 2);
  IntArrayRef batch_tensor2(x2.sizes().data(), dim2 - 2);
  std::vector<int64_t> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);
  std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
  tensor1_expand_size.insert(tensor1_expand_size.end(), {r1, c1});
  std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
  tensor2_expand_size.insert(tensor2_expand_size.end(), {r2, c2});

  const int64_t expand_batch_product = c10::multiply_integers(expand_batch_portion);
  std::vector<int64_t> tensor1_view{expand_batch_product, r1, c1};
  std::vector<int64_t> tensor2_view{expand_batch_product, r2, c2};

  std::vector<int64_t> output_shape(expand_batch_portion);
  output_shape.insert(output_shape.end(), {r1, r2});
  Tensor result = at::empty(output_shape, x1.options());

  NormOpBlock norm_op_block = ^NormOpFn(cachedGraph, x1Tensor, x2Tensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();

    MPSGraphTensor* inputBroadcast = [mpsGraph broadcastTensor:x1Tensor
                                                       toShape:getMPSShape(tensor1_expand_size)
                                                          name:nil];
    MPSGraphTensor* inputBroadcastReshape = [mpsGraph reshapeTensor:inputBroadcast
                                                          withShape:getMPSShape(tensor1_view)
                                                               name:nil];

    MPSGraphTensor* otherBroadcast = [mpsGraph broadcastTensor:x2Tensor
                                                       toShape:getMPSShape(tensor2_expand_size)
                                                          name:nil];
    MPSGraphTensor* otherBroadcastReshape = [mpsGraph reshapeTensor:otherBroadcast
                                                          withShape:getMPSShape(tensor2_view)
                                                               name:nil];

    NSMutableArray<MPSGraphTensor*>* inputArray = [NSMutableArray arrayWithCapacity:tensor1_view[1]];
    NSMutableArray<MPSGraphTensor*>* otherArray = [NSMutableArray arrayWithCapacity:tensor2_view[1]];

    for (const auto i : c10::irange(tensor2_view[1])) {
      inputArray[i] = inputBroadcastReshape;
    }

    for (const auto i : c10::irange(tensor1_view[1])) {
      otherArray[i] = otherBroadcastReshape;
    }

    MPSGraphTensor* inputTensorReshaped = [mpsGraph concatTensors:inputArray dimension:1 interleave:YES name:nil];
    MPSGraphTensor* otherTensorReshaped = [mpsGraph concatTensors:otherArray dimension:1 interleave:NO name:nil];

    MPSGraphTensor* inputTensorPNorm = [mpsGraph subtractionWithPrimaryTensor:inputTensorReshaped
                                                              secondaryTensor:otherTensorReshaped
                                                                         name:nil];
    return inputTensorPNorm;
  };

  c10::optional<IntArrayRef> inputBroadcastSize =
      c10::make_optional(makeArrayRef(tensor1_view.data(), tensor1_view.size()));
  impl_func_norm_mps(x1,
                     x2,
                     OptionalScalarRef(p),
                     makeArrayRef<int64_t>(2),
                     false,
                     c10::nullopt,
                     result,
                     /*cdist=*/true,
                     inputBroadcastSize,
                     norm_op_block);
  return result;
}

Tensor var_mps(const Tensor& input_t,
               at::OptionalIntArrayRef dim,
               const c10::optional<Scalar>& correction,
               bool keepdim) {
  return mps::std_var_common_impl_mps(input_t, dim, correction, keepdim, mps::STANDARD_VARIANCE);
}

Tensor std_mps(const Tensor& input_t,
               at::OptionalIntArrayRef dim,
               const c10::optional<Scalar>& correction,
               bool keepdim) {
  return mps::std_var_common_impl_mps(input_t, dim, correction, keepdim, mps::STANDARD_DEVIATION);
}

TORCH_IMPL_FUNC(any_out_mps)
(const Tensor& input_t, int64_t dim, bool keepdim, const Tensor& output_t) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (output_t.numel() == 0 || input_t.numel() == 0) {
    return;
  }

  bool macOS13_3_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
  MPS_CHECK_INT64_OP_SUPPORTED(input_t, macOS13_3_plus, "any_out");

  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  native::zero_numel_check_dims(input_t, dim_, "any()");

  // Calculate the output shape according to keepdim=True
  // If there is no dim argument, the input shape is flattened
  IntArrayRef input_shape = input_t.sizes();
  int64_t num_input_dims = input_shape.size();
  NSMutableArray<NSNumber*>* apparent_out_shape = nil;
  apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
  for (const auto i : c10::irange(num_input_dims)) {
    apparent_out_shape[i] = dim_ == i ? @1 : [NSNumber numberWithInt:input_shape[i]];
  }

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    MPSShape* input_t_shape = getMPSShape(input_t);
    string key = string("any_out_mps:") + getMPSShapeString(input_t_shape) + ":" + to_string(dim_) + ":" +
        getMPSTypeString(input_t);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSDataType input_type = getMPSDataType(input_t);
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_type, input_t_shape);

      MPSGraphTensor* castInputTensor =
          castToIHFTypes(mpsGraph, inputTensor, input_t, /*includesInt64=*/macOS13_3_plus);
      MPSGraphTensor* castOutputTensor = [mpsGraph reductionOrWithTensor:castInputTensor axis:dim_ name:nil];
      MPSGraphTensor* outputTensor = castOutputTensor;
      if (MPSDataTypeBool != [castOutputTensor dataType]) {
        outputTensor = [mpsGraph castTensor:castOutputTensor toType:MPSDataTypeBool name:@"outputTensor"];
      }
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t, apparent_out_shape);
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(any_all_out_mps)(const Tensor& input_t, const Tensor& output_t) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;
  if (input_t.numel() == 0) {
    output_t.zero_();
    return;
  } else if (input_t.numel() == 1) {
    output_t.copy_(input_t.view_as(output_t).to(at::kBool));
    return;
  } else if (output_t.numel() == 0) {
    return;
  }

  bool macOS13_3_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
  MPS_CHECK_INT64_OP_SUPPORTED(input_t, macOS13_3_plus, "any_all_out");

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    MPSShape* input_t_shape = getMPSShape(input_t);
    string key = string("any_all_out_mps:") + getMPSShapeString(input_t_shape) + ":" + getMPSTypeString(input_t);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSDataType input_type = getMPSDataType(input_t);
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_type, input_t_shape);
      MPSGraphTensor* castInputTensor =
          castToIHFTypes(mpsGraph, inputTensor, input_t, /*includesInt64=*/macOS13_3_plus);
      MPSGraphTensor* castOutputTensor = [mpsGraph reductionOrWithTensor:castInputTensor axes:nil name:nil];

      MPSGraphTensor* outputTensor = castOutputTensor;
      if (getMPSDataType(output_t) != [castOutputTensor dataType]) {
        outputTensor = castMPSTensor(mpsGraph, castOutputTensor, output_t.scalar_type());
      }
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t);
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(all_out_mps)
(const Tensor& input_t, int64_t dim, bool keepdim, const Tensor& output_t) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (output_t.numel() == 0 || input_t.numel() == 0) {
    return;
  }

  bool macOS13_3_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
  MPS_CHECK_INT64_OP_SUPPORTED(input_t, macOS13_3_plus, "all_out");

  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  native::zero_numel_check_dims(input_t, dim_, "all()");

  // Calculate the output shape according to keepdim=True
  // If there is no dim argument, the input shape is flattened
  IntArrayRef input_shape = input_t.sizes();
  int64_t num_input_dims = input_shape.size();
  NSMutableArray<NSNumber*>* apparent_out_shape = nil;
  apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
  for (const auto i : c10::irange(num_input_dims)) {
    apparent_out_shape[i] = dim_ == i ? @1 : [NSNumber numberWithInt:input_shape[i]];
  }

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    MPSShape* input_t_shape = getMPSShape(input_t);
    string key = string("all_out_mps:") + getMPSShapeString(input_t_shape) + ":" + to_string(dim_) + ":" +
        getMPSTypeString(input_t);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSDataType input_type = getMPSDataType(input_t);
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_type, input_t_shape);
      MPSGraphTensor* castInputTensor =
          castToIHFTypes(mpsGraph, inputTensor, input_t, /*includesInt64=*/macOS13_3_plus);
      MPSGraphTensor* castOutputTensor = [mpsGraph reductionAndWithTensor:castInputTensor axis:dim_ name:nil];
      MPSGraphTensor* outputTensor = castOutputTensor;
      if (MPSDataTypeBool != [castOutputTensor dataType]) {
        outputTensor = castMPSTensor(mpsGraph, castOutputTensor, MPSDataTypeBool);
      }
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t, apparent_out_shape);
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(all_all_out_mps)(const Tensor& input_t, const Tensor& output_t) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;
  if (output_t.numel() == 0 || input_t.numel() == 0) {
    // in line with cpu behaviour and numpy, an empty tensor should return true.
    // specifying ones forces the output to be true for this case.
    output_t.fill_(1);
    return;
  }

  bool macOS13_3_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
  MPS_CHECK_INT64_OP_SUPPORTED(input_t, macOS13_3_plus, "all_all_out");

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    MPSShape* input_t_shape = getMPSShape(input_t);
    string key = string("all_all_out_mps:") + getMPSShapeString(input_t_shape) + ":" + getMPSTypeString(input_t);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSDataType input_type = getMPSDataType(input_t);
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_type, input_t_shape);
      MPSGraphTensor* castInputTensor =
          castToIHFTypes(mpsGraph, inputTensor, input_t, /*includesInt64=*/macOS13_3_plus);
      MPSGraphTensor* castOutputTensor = [mpsGraph reductionAndWithTensor:castInputTensor axes:nil name:nil];
      MPSGraphTensor* outputTensor = castOutputTensor;
      if (MPSDataTypeBool != [castOutputTensor dataType]) {
        outputTensor = castMPSTensor(mpsGraph, castOutputTensor, MPSDataTypeBool);
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t);
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

//-----------------------------------------------------------------------
// Min and max functions

// Max entire tensor into scalar result
Tensor max_mps(const Tensor& input_t) {
  return mps::min_max_mps_impl(input_t, mps::MPSReductionType::MAX, "max_mps");
}

// Min entire tensor into scalar result
Tensor min_mps(const Tensor& input_t) {
  return mps::min_max_mps_impl(input_t, mps::MPSReductionType::MIN, "min_mps");
}

// Max out with dim
TORCH_IMPL_FUNC(max_out_mps)
(const Tensor& input_t, int64_t dim, bool keepdim, const Tensor& output_t, const Tensor& indices_t) {
  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  native::zero_numel_check_dims(input_t, dim_, "max()");

  mps::min_max_out_mps(input_t, dim, keepdim, output_t, indices_t, mps::MPSReductionType::MAX, "max_out_mps");
}

// Min out with dim
TORCH_IMPL_FUNC(min_out_mps)
(const Tensor& input_t, int64_t dim, bool keepdim, const Tensor& output_t, const Tensor& indices_t) {
  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  native::zero_numel_check_dims(input_t, dim_, "min()");

  mps::min_max_out_mps(input_t, dim, keepdim, output_t, indices_t, mps::MPSReductionType::MIN, "min_out_mps");
}

TORCH_IMPL_FUNC(argmax_out_mps)
(const Tensor& input_t, c10::optional<int64_t> dim, bool keepdim, const Tensor& output_t) {
  mps::argmax_argmin_out_mps(input_t, dim, keepdim, output_t, mps::MPSReductionType::MAX, "argmax_out_mps");
}

TORCH_IMPL_FUNC(argmin_out_mps)
(const Tensor& input_t, c10::optional<int64_t> dim, bool keepdim, const Tensor& output_t) {
  mps::argmax_argmin_out_mps(input_t, dim, keepdim, output_t, mps::MPSReductionType::MIN, "argmin_out_mps");
}

// Max with dim
static std::tuple<Tensor, Tensor> max_mps(const Tensor& input_t, int64_t dim, bool keepdim) {
  return mps::min_max_mps_impl(input_t, dim, keepdim, mps::MPSReductionType::MAX, "max_mps");
}

// Min with dim
static std::tuple<Tensor, Tensor> min_mps(const Tensor& input_t, int64_t dim, bool keepdim) {
  return mps::min_max_mps_impl(input_t, dim, keepdim, mps::MPSReductionType::MIN, "min_mps");
}

// Median of entire tensor into scalar result
Tensor median_mps(const Tensor& input_t) {
  if (!is_macos_13_or_newer()) {
    TORCH_WARN_ONCE("MPS: median op is supported natively starting from macOS 13.0. ",
                    "Falling back on CPU. This may have performance implications.");
    return at::median(input_t.to("cpu"));
  }

  using namespace mps;

  bool macOS13_3_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
  MPS_CHECK_INT64_OP_SUPPORTED(input_t, macOS13_3_plus, "median");

  using CachedGraph = MPSUnaryCachedGraph;

  IntArrayRef input_shape = input_t.sizes();

  // calculate total no. of elements in the input tensor to reduce it to one dimension
  NSMutableArray<NSNumber*>* apparent_input_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:1];
  int64_t num_in_elements = c10::multiply_integers(input_shape);

  apparent_input_shape[0] = [NSNumber numberWithInt:num_in_elements];

  Tensor output_t = at::empty({}, input_t.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);

  if (output_t.numel() == 0 || num_in_elements == 0) {
    return output_t;
  }

  @autoreleasepool {
    string key = "median_mps:" + mps::getMPSTypeString(input_t) + mps::getTensorsStringKey(input_t);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_t);
      MPSGraphTensor* castInputTensor =
          castToIHFTypes(mpsGraph, inputTensor, input_t, /*includesInt64=*/macOS13_3_plus);

      auto reshapedTensor = [mpsGraph reshapeTensor:castInputTensor withShape:@[ @-1 ] name:nil];
      auto sortedTensor = [mpsGraph sortWithTensor:reshapedTensor axis:((NSUInteger)(int)0)name:nil];
      auto outputTensor = [mpsGraph sliceTensor:sortedTensor
                                      dimension:0
                                          start:((NSUInteger)(int)((num_in_elements + 1) / 2) - 1)
                                         length:1
                                           name:nil];

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

static void median_out_mps(const Tensor& input_t,
                           int64_t dim,
                           bool keepdim,
                           const Tensor& output_t,
                           const Tensor& indices_t,
                           const std::string& func_name) {
  using namespace mps;

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

  bool macOS13_3_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
  MPS_CHECK_INT64_OP_SUPPORTED(input_t, macOS13_3_plus, "median_out");

  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());

  // Calculate the output shape according to keepdim=True
  // If there is no dim argument, the input shape is flattened
  IntArrayRef input_shape = input_t.sizes();
  int64_t num_input_dims = input_shape.size();
  NSMutableArray<NSNumber*>* apparent_out_shape = nil;

  apparent_out_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
  for (const int i : c10::irange(num_input_dims)) {
    apparent_out_shape[i] = dim_ == i ? @1 : [NSNumber numberWithInt:input_shape[i]];
  }
  int dim_total_elements = input_shape[dim_];

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    string key =
        func_name + ":" + to_string(dim_) + ":" + getTensorsStringKey(input_t) + ":" + getTensorsStringKey(indices_t);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_t);
      MPSGraphTensor* castInputTensor =
          castToIHFTypes(mpsGraph, inputTensor, input_t, /*includesInt64=*/macOS13_3_plus);

      MPSGraphTensor* sortedTensor = [mpsGraph sortWithTensor:castInputTensor axis:((NSUInteger)(int)dim_)name:nil];

      MPSGraphTensor* outputTensor = [mpsGraph sliceTensor:sortedTensor
                                                 dimension:dim_
                                                     start:((NSUInteger)(int)((dim_total_elements + 1) / 2) - 1)
                                                    length:1
                                                      name:nil];
      MPSGraphTensor* argreduceOutTensor = nil;
      argreduceOutTensor = [mpsGraph argSortWithTensor:castInputTensor axis:(NSInteger)dim_ name:@"argmax_out"];
      MPSGraphTensor* argOutputTensor = [mpsGraph sliceTensor:argreduceOutTensor
                                                    dimension:dim_
                                                        start:((NSUInteger)(int)((dim_total_elements + 1) / 2) - 1)
                                                       length:1
                                                         name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
      newCachedGraph->indicesTensor_ = argOutputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output_t, apparent_out_shape);
    auto indicesPlaceholder = Placeholder(cachedGraph->indicesTensor_, indices_t, apparent_out_shape);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    auto results = dictionaryFromPlaceholders(outputPlaceholder, indicesPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}

// in case mps sortWithTensor do not supported on macOS
static std::tuple<Tensor&, Tensor&> median_from_cpu(const Tensor& self,
                                                    int64_t dim,
                                                    bool keepdim,
                                                    Tensor& valuesI,
                                                    Tensor& indicesI,
                                                    IntArrayRef vec_out_shape,
                                                    IntArrayRef vec_apparent_out_shape) {
  Tensor values;
  Tensor indices;
  if (!keepdim) {
    values = at::empty({vec_out_shape}, self.options());
    indices = at::empty({vec_out_shape}, self.options().dtype(kLong));
  } else {
    values = at::empty({vec_apparent_out_shape}, self.options());
    indices = at::empty({vec_apparent_out_shape}, self.options().dtype(kLong));
  }
  at::median_out(values, indices, self, dim, keepdim);

  valuesI.copy_(values);
  indicesI.copy_(indices);
  return std::forward_as_tuple(valuesI, indicesI);
}

TORCH_API ::std::tuple<at::Tensor&, at::Tensor&> median_out_mps(const at::Tensor& input_t,
                                                                int64_t dim,
                                                                bool keepdim,
                                                                at::Tensor& values,
                                                                at::Tensor& indices) {
  bool macOS13_3_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
  MPS_CHECK_INT64_OP_SUPPORTED(input_t, macOS13_3_plus, "median_out");

  int64_t dim_ = maybe_wrap_dim(dim, input_t.dim());
  native::zero_numel_check_dims(input_t, dim_, "max()");

  // Calculate the output shape according to keepdim=True
  // If there is no dim argument, the input shape is flattened
  IntArrayRef input_shape = input_t.sizes();
  int64_t num_input_dims = input_shape.size();
  NSMutableArray<NSNumber*>* apparent_out_shape = nil;
  // Use this if keepdim is false
  int64_t num_output_dims = num_input_dims - 1 < 0 ? 0 : num_input_dims - 1;

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

  if (!keepdim) {
    values =
        at::empty(IntArrayRef(vec_out_shape), input_t.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
    indices = at::empty(IntArrayRef(vec_out_shape), ScalarType::Long, c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
  } else {
    values = at::empty(
        IntArrayRef(vec_apparent_out_shape), input_t.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
    indices = at::empty(
        IntArrayRef(vec_apparent_out_shape), ScalarType::Long, c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
  }

  if (values.numel() == 0 || input_t.numel() == 0) {
    return std::tuple<Tensor&, Tensor&>{values, indices};
  }

  if (!is_macos_13_or_newer()) {
    TORCH_WARN_ONCE("MPS: median op is supported natively starting from macOS 13.0.",
                    "Falling back on CPU. This may have performance implications.");
    return median_from_cpu(input_t.to("cpu"),
                           dim,
                           keepdim,
                           values,
                           indices,
                           IntArrayRef(vec_out_shape),
                           IntArrayRef(vec_apparent_out_shape));
  }

  median_out_mps(input_t, dim, keepdim, values, indices, "median_out_mps");

  return std::tuple<Tensor&, Tensor&>{values, indices};
}

std::tuple<Tensor, Tensor> std_mean_mps(const Tensor& self,
                                        at::OptionalIntArrayRef dim,
                                        const c10::optional<Scalar>& correction,
                                        bool keepdim) {
  // TODO: Refactor it into a proper std_var_mean composite function
  auto std = std_mps(self, dim, correction, keepdim);
  auto mean = at::empty(std.sizes(), self.scalar_type(), c10::nullopt, kMPS, c10::nullopt, MemoryFormat::Contiguous);
  mps::reduction_out_mps(self, dim, keepdim, c10::nullopt, mean, mps::MPSReductionType::MEAN, "mean_out_mps");
  return {std, mean};
}

std::tuple<Tensor, Tensor> var_mean_mps(const Tensor& self,
                                        at::OptionalIntArrayRef dim,
                                        const c10::optional<Scalar>& correction,
                                        bool keepdim) {
  // TODO: Refactor it into a proper std_var_mean composite function
  auto var = var_mps(self, dim, correction, keepdim);
  auto mean = at::empty(var.sizes(), self.scalar_type(), c10::nullopt, kMPS, c10::nullopt, MemoryFormat::Contiguous);
  mps::reduction_out_mps(self, dim, keepdim, c10::nullopt, mean, mps::MPSReductionType::MEAN, "mean_out_mps");
  return {var, mean};
}

} // namespace at::native
