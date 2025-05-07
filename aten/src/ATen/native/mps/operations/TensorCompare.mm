//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/clamp_max_native.h>
#include <ATen/ops/clamp_min_native.h>
#include <ATen/ops/clamp_native.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/isin_native.h>
#include <ATen/ops/nan_to_num_native.h>
#include <ATen/ops/ones_like_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/where_native.h>
#endif

namespace at::native {
namespace mps {

struct CachedGraph : public MPSCachedGraph {
  CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor *inputTensor = nil, *outputTensor = nil;
  MPSGraphTensor *minTensor = nil, *maxTensor = nil;
};

static void clamp_mps_graph(CachedGraph* cachedGraph,
                            const Tensor& input_tensor,
                            const at::ScalarType min_type,
                            const at::ScalarType max_type,
                            const at::ScalarType result_type) {
  MPSGraph* mpsGraph = cachedGraph->graph();

  cachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_tensor);

  auto minTensor = cachedGraph->minTensor;
  auto maxTensor = cachedGraph->maxTensor;
  auto inputTensor = cachedGraph->inputTensor;

  if (minTensor && min_type != result_type) {
    minTensor = castMPSTensor(mpsGraph, minTensor, result_type);
  }
  if (maxTensor && max_type != result_type) {
    maxTensor = castMPSTensor(mpsGraph, maxTensor, result_type);
  }
  if (input_tensor.scalar_type() != result_type) {
    inputTensor = castMPSTensor(mpsGraph, inputTensor, result_type);
  }
  if (c10::isIntegralType(result_type, /*includeBool=*/true)) {
    if (minTensor && maxTensor) {
      cachedGraph->outputTensor = [mpsGraph clampWithTensor:inputTensor
                                             minValueTensor:minTensor
                                             maxValueTensor:maxTensor
                                                       name:nil];
    } else if (maxTensor) {
      cachedGraph->outputTensor = [mpsGraph minimumWithPrimaryTensor:inputTensor secondaryTensor:maxTensor name:nil];
    } else if (minTensor) {
      cachedGraph->outputTensor = [mpsGraph maximumWithPrimaryTensor:inputTensor secondaryTensor:minTensor name:nil];
    }
    return;
  }
  // clampWithTensor doesn't propagate NaN through so simulate it as composition of
  // maximumWithNaNPropagationWithPrimaryTensor and minimumWithNaNPropagationWithPrimaryTensor
  auto outputTensor = inputTensor;
  if (minTensor) {
    outputTensor = [mpsGraph maximumWithNaNPropagationWithPrimaryTensor:outputTensor
                                                        secondaryTensor:minTensor
                                                                   name:nil];
  }
  if (maxTensor) {
    outputTensor = [mpsGraph minimumWithNaNPropagationWithPrimaryTensor:outputTensor
                                                        secondaryTensor:maxTensor
                                                                   name:nil];
  }
  cachedGraph->outputTensor = outputTensor;
}

static void check_min_max_dims(const OptionalTensorRef clamp_opt, const Tensor& input_t, std::string op_name) {
  if (!clamp_opt->is_same_size(input_t)) {
    auto num_clamp_dims = clamp_opt->dim();
    auto num_input_dims = input_t.dim();

    auto clamp_shape = clamp_opt->sizes();
    auto input_shape = input_t.sizes();

    TORCH_CHECK(num_clamp_dims <= num_input_dims,
                op_name + ": clamp tensor number of dims must not be greater than that of input tensor")

    for (int i = 0; i < num_clamp_dims; i++)
      // One of the indices is allowed to be 1; will be handled by broadcast
      TORCH_CHECK(clamp_shape[num_clamp_dims - 1 - i] == input_shape[num_input_dims - 1 - i] ||
                      clamp_shape[num_clamp_dims - 1 - i] == 1 || input_shape[num_input_dims - 1 - i] == 1,
                  op_name + ": clamp tensor trailing shape must match input tensor")
  }
}

static void fill_new_shape(int64_t num_input_dims,
                           int64_t num_clamp_dims,
                           int64_t* new_shape,
                           IntArrayRef clamp_shape) {
  // Extend the shape with ones to the left
  int clamp_idx = 0;
  for (int i = 0; i < num_input_dims; i++) {
    if (i < num_input_dims - num_clamp_dims)
      new_shape[i] = 1;
    else {
      new_shape[i] = clamp_shape[clamp_idx];
      clamp_idx++;
    }
  }
}

static void clamp_tensor_out_mps(const Tensor& input_t,
                                 const OptionalTensorRef min_opt,
                                 const OptionalTensorRef max_opt,
                                 const Tensor& output_t,
                                 std::string op_name) {
  const bool has_min = (min_opt.has_value() && min_opt->defined());
  const bool has_max = (max_opt.has_value() && max_opt->defined());

  TORCH_CHECK(has_min || has_max, op_name + ": either min, max or both tensors must be defined")
  if (has_min)
    check_min_max_dims(min_opt, input_t, op_name);

  if (has_max)
    check_min_max_dims(max_opt, input_t, op_name);

  if (output_t.numel() == 0)
    return;

  auto result_type = output_t.scalar_type();

  IntArrayRef new_min_shape;
  IntArrayRef new_max_shape;

  auto num_min_dims = min_opt->dim();
  auto num_max_dims = max_opt->dim();
  auto num_input_dims = input_t.dim();

  std::vector<int64_t> new_min_arr(num_input_dims);
  std::vector<int64_t> new_max_arr(num_input_dims);

  if (has_min && num_min_dims < num_input_dims) {
    fill_new_shape(num_input_dims, num_min_dims, new_min_arr.data(), min_opt->sizes());
    new_min_shape = IntArrayRef(new_min_arr);
  }

  if (has_max && num_max_dims < num_input_dims) {
    fill_new_shape(num_input_dims, num_max_dims, new_max_arr.data(), max_opt->sizes());
    new_max_shape = IntArrayRef(new_max_arr);
  }

  Tensor min_opt_tensor;
  Tensor max_opt_tensor;

  if (has_min) {
    min_opt_tensor = (num_min_dims < num_input_dims) ? (*min_opt).view(new_min_shape) : *min_opt;
  }
  if (has_max) {
    max_opt_tensor = (num_max_dims < num_input_dims) ? (*max_opt).view(new_max_shape) : *max_opt;
  }

  @autoreleasepool {
    // the optional min/max refs could affect how we build the cached graph

    auto tensor_key = has_min
        ? (has_max ? getTensorsStringKey({input_t, min_opt_tensor, max_opt_tensor})
                   : getTensorsStringKey({input_t, min_opt_tensor}))
        : (has_max ? getTensorsStringKey({input_t, max_opt_tensor}) : getTensorsStringKey({input_t}));

    std::string key = op_name + (has_min ? "_min" : "") + (has_max ? "_max" : "") + "_tensor" + tensor_key;
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      if (has_min) {
        newCachedGraph->minTensor = mpsGraphRankedPlaceHolder(mpsGraph, min_opt_tensor);
      }
      if (has_max) {
        newCachedGraph->maxTensor = mpsGraphRankedPlaceHolder(mpsGraph, max_opt_tensor);
        ;
      }

      clamp_mps_graph(newCachedGraph, input_t, min_opt_tensor.scalar_type(), max_opt_tensor.scalar_type(), result_type);
    });

    bool gatherTensorData = true;
    if (!output_t.is_contiguous() || output_t.is_view()) {
      gatherTensorData = false;
    }

    auto inputPlaceholder =
        Placeholder(cachedGraph->inputTensor, input_t, /*mpsShape=*/nil, /*gatherTensorData=*/gatherTensorData);
    auto outputPlaceholder =
        Placeholder(cachedGraph->outputTensor, output_t, /*mpsShape=*/nil, /*gatherTensorData=*/false);

    NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    if (has_min) {
      min_opt_tensor =
          gatherTensorData && !min_opt_tensor.is_contiguous() ? min_opt_tensor.contiguous() : min_opt_tensor;
      auto minPlaceholder =
          Placeholder(cachedGraph->minTensor, min_opt_tensor, /*mpsShape=*/nil, /*gatherTensorData=*/gatherTensorData);
      feeds[minPlaceholder.getMPSGraphTensor()] = minPlaceholder.getMPSGraphTensorData();
    }
    if (has_max) {
      max_opt_tensor =
          gatherTensorData && !max_opt_tensor.is_contiguous() ? max_opt_tensor.contiguous() : max_opt_tensor;
      auto maxPlaceholder =
          Placeholder(cachedGraph->maxTensor, max_opt_tensor, /*mpsShape=*/nil, /*gatherTensorData=*/gatherTensorData);
      feeds[maxPlaceholder.getMPSGraphTensor()] = maxPlaceholder.getMPSGraphTensorData();
    }

    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

static void clamp_scalar_out_mps(const Tensor& input_t,
                                 const OptionalScalarRef min_opt,
                                 const OptionalScalarRef max_opt,
                                 const Tensor& output_t,
                                 std::string op_name) {
  using scalar_t = double;

  const bool has_min = (min_opt.has_value());
  const bool has_max = (max_opt.has_value());
  TORCH_CHECK(has_min || has_max, op_name + ": either min, max or both scalars must be defined")

  scalar_t min_scalar = std::numeric_limits<scalar_t>::infinity();
  scalar_t max_scalar = -std::numeric_limits<scalar_t>::infinity();

  if (has_min)
    min_scalar = min_opt.get().to<scalar_t>();
  if (has_max)
    max_scalar = max_opt.get().to<scalar_t>();

  if (output_t.numel() == 0)
    return;

  auto result_type = output_t.scalar_type();

  @autoreleasepool {
    // the optional min/max refs could affect how we build the cached graph
    std::string key = op_name + (has_min ? ("_min:" + std::to_string(min_scalar)) : "") +
        (has_max ? ("_max:" + std::to_string(max_scalar)) : "") + "_scalar:" + getTensorsStringKey({input_t});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      if (has_min)
        newCachedGraph->minTensor = [mpsGraph constantWithScalar:min_scalar
                                                           shape:mps::getMPSShape(input_t)
                                                        dataType:mps::getMPSScalarType(result_type)];
      if (has_max)
        newCachedGraph->maxTensor = [mpsGraph constantWithScalar:max_scalar
                                                           shape:mps::getMPSShape(input_t)
                                                        dataType:mps::getMPSScalarType(result_type)];

      clamp_mps_graph(newCachedGraph, input_t, result_type, result_type, result_type);
    });

    bool gatherTensorData = true;
    if (!output_t.is_contiguous() || output_t.is_view()) {
      gatherTensorData = false;
    }

    auto inputPlaceholder =
        Placeholder(cachedGraph->inputTensor, input_t, /*mpsShape=*/nil, /*gatherTensorData=*/gatherTensorData);
    auto outputPlaceholder =
        Placeholder(cachedGraph->outputTensor, output_t, /*mpsShape=*/nil, /*gatherTensorData=*/false);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

static void isin_Tensor_Tensor_out_mps(const Tensor& elements,
                                       const Tensor& test_elements,
                                       bool assume_unique,
                                       bool invert,
                                       const Tensor& out,
                                       std::string op_name) {
  if (elements.numel() == 0) {
    return;
  }

  if (test_elements.numel() == 0) {
    if (invert) {
      auto ones = ones_like(out);
      out.copy_(ones);
    } else {
      auto zeros = zeros_like(out);
      out.copy_(zeros);
    }
    return;
  }

  const auto common_type = at::result_type(elements, test_elements);
  TORCH_CHECK(elements.is_mps() && test_elements.is_mps());
  TORCH_CHECK(is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS) || supportedFloatingType(common_type),
              "isin_Tensor_Tensor_out only works on floating types on MPS for pre MacOS_14_0. Received dtype: ",
              common_type);

  @autoreleasepool {
    std::string key = op_name + getTensorsStringKey({elements, test_elements}) + std::to_string(invert);

    auto cachedGraph = LookUpOrCreateCachedGraph<MPSBinaryCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->inputTensor_ = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(elements.scalar_type()));
      newCachedGraph->otherTensor_ = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(test_elements.scalar_type()));

      // Cast to common type
      auto inputTensor = castMPSTensor(mpsGraph, newCachedGraph->inputTensor_, common_type);
      auto otherTensor = castMPSTensor(mpsGraph, newCachedGraph->otherTensor_, common_type);

      MPSShape* outputShape = getMPSShape(out);

      MPSGraphTensor* input_flattened = [mpsGraph reshapeTensor:inputTensor withShape:@[ @-1, @1 ] name:nil];
      MPSGraphTensor* other_flattened = [mpsGraph reshapeTensor:otherTensor withShape:@[ @1, @-1 ] name:nil];
      MPSGraphTensor* isInTensor = [mpsGraph equalWithPrimaryTensor:input_flattened
                                                    secondaryTensor:other_flattened
                                                               name:nil];
      MPSGraphTensor* output = [mpsGraph reductionOrWithTensor:isInTensor axis:1 name:nil];
      output = [mpsGraph reshapeTensor:output withShape:outputShape name:nil];

      if (invert) {
        output = [mpsGraph notWithTensor:output name:nil];
      }
      newCachedGraph->outputTensor_ = output;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, elements);
    auto otherPlaceholder = Placeholder(cachedGraph->otherTensor_, test_elements);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder, otherPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

static void is_posneginf_helper(TensorIteratorBase& iter, bool is_neg) {
  const auto& self = iter.input(0);
  auto& out = iter.output(0);
  @autoreleasepool {
    auto cachedGraph = LookUpOrCreateCachedGraph<MPSUnaryCachedGraph>(
        __func__ + std::to_string(is_neg) + getTensorsStringKey(self), [&](auto mpsGraph, auto newCachedGraph) {
          auto infTensor = [mpsGraph constantWithScalar:is_neg ? -std::numeric_limits<float>::infinity()
                                                               : std::numeric_limits<float>::infinity()
                                               dataType:getMPSScalarType(self)];
          newCachedGraph->inputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, self);
          newCachedGraph->outputTensor_ = [mpsGraph equalWithPrimaryTensor:newCachedGraph->inputTensor_
                                                           secondaryTensor:infTensor
                                                                      name:nil];
        });
    auto selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);
    runMPSGraph(
        getCurrentMPSStream(), cachedGraph->graph(), dictionaryFromPlaceholders(selfPlaceholder), outputPlaceholder);
  }
}
} // namespace mps

// APIs exposed to at::native scope
TORCH_IMPL_FUNC(clamp_Tensor_out_mps)
(const Tensor& input_t, const OptionalTensorRef min, const OptionalTensorRef max, const Tensor& output_t) {
  mps::clamp_tensor_out_mps(input_t, min, max, output_t, __func__);
}

TORCH_IMPL_FUNC(clamp_out_mps)
(const Tensor& input_t, const OptionalScalarRef min, const OptionalScalarRef max, const Tensor& output_t) {
  mps::clamp_scalar_out_mps(input_t, min, max, const_cast<Tensor&>(output_t), "clamp_out_mps");
}

TORCH_IMPL_FUNC(clamp_min_Tensor_out_mps)
(const Tensor& input_t, const Tensor& min, const Tensor& output_t) {
  mps::clamp_tensor_out_mps(input_t, min, at::OptionalTensorRef(), output_t, __func__);
}

TORCH_IMPL_FUNC(clamp_min_out_mps)
(const Tensor& input_t, const Scalar& min, const Tensor& output_t) {
  mps::clamp_scalar_out_mps(input_t, min, at::OptionalScalarRef(), output_t, __func__);
}

TORCH_IMPL_FUNC(clamp_max_Tensor_out_mps)
(const Tensor& input_t, const Tensor& max, const Tensor& output_t) {
  mps::clamp_tensor_out_mps(input_t, at::OptionalTensorRef(), max, output_t, __func__);
}

TORCH_IMPL_FUNC(clamp_max_out_mps)
(const Tensor& input_t, const Scalar& max, const Tensor& output_t) {
  mps::clamp_scalar_out_mps(input_t, at::OptionalScalarRef(), max, output_t, __func__);
}

TORCH_IMPL_FUNC(isin_Tensor_Tensor_out_mps)
(const Tensor& elements, const Tensor& test_elements, bool assume_unique, bool invert, const Tensor& out) {
  mps::isin_Tensor_Tensor_out_mps(elements, test_elements, assume_unique, invert, out, __func__);
}

static void where_kernel_mps(TensorIterator& iter) {
  const auto& condition = iter.input(0);
  const auto& self = iter.input(1);
  const auto& other = iter.input(2);
  auto& out = iter.output(0);
  TORCH_CHECK(condition.device() == self.device() && self.device() == other.device(),
              "Expected all tensors to be on the same device, but found at least two devices.");
  TORCH_CHECK(self.dtype() == other.dtype(), "expected scalar type ", self.dtype(), " but found ", other.dtype());

  if (condition.scalar_type() == ScalarType::Byte) {
    TORCH_WARN_ONCE(
        "where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead.");
  } else {
    TORCH_CHECK(condition.scalar_type() == ScalarType::Bool,
                "where expected condition to be a boolean tensor, but got a tensor with dtype ",
                condition.scalar_type());
  }
  Tensor cond_bool = condition.scalar_type() == ScalarType::Byte ? condition.to(ScalarType::Bool) : condition;

  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();

  // Empty output
  if (out.numel() == 0) {
    return;
  }

  Tensor out_;
  if (needsGather(out)) {
    out_ = out.contiguous();
  }

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* conditionTensor_ = nil;
    MPSGraphTensor* selfTensor_ = nil;
    MPSGraphTensor* otherTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSDataType conditionDataType = getMPSScalarType(condition.scalar_type());
  MPSDataType selfDataType = getMPSScalarType(self.scalar_type());
  MPSDataType otherDataType = getMPSScalarType(other.scalar_type());

  @autoreleasepool {
    std::string key = "where_self_out_mps:" + getTensorsStringKey({cond_bool, self, other});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* conditionTensor = mpsGraphRankedPlaceHolder(mpsGraph, conditionDataType, getMPSShape(cond_bool));
      MPSGraphTensor* selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, selfDataType, getMPSShape(self));
      MPSGraphTensor* otherTensor = mpsGraphRankedPlaceHolder(mpsGraph, otherDataType, getMPSShape(other));

      MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:conditionTensor
                                                     truePredicateTensor:selfTensor
                                                    falsePredicateTensor:otherTensor
                                                                    name:nil];

      newCachedGraph->conditionTensor_ = conditionTensor;
      newCachedGraph->selfTensor_ = selfTensor;
      newCachedGraph->otherTensor_ = otherTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder conditionPlaceholder = Placeholder(
        cachedGraph->conditionTensor_, cond_bool, /*mpsShape=*/nullptr, /*gatherTensorData=*/true, conditionDataType);
    Placeholder selfPlaceholder =
        Placeholder(cachedGraph->selfTensor_, self, /*mpsShape=*/nullptr, /*gatherTensorData=*/true, selfDataType);
    Placeholder otherPlaceholder =
        Placeholder(cachedGraph->otherTensor_, other, /*mpsShape=*/nullptr, /*gatherTensorData=*/true, otherDataType);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_,
                                                needsGather(out) ? out_ : out,
                                                /*mpsShape=*/nullptr,
                                                /*gatherTensorData=*/needsGather(out),
                                                getMPSScalarType(out.scalar_type()));

    auto feeds = dictionaryFromPlaceholders(conditionPlaceholder, selfPlaceholder, otherPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (needsGather(out)) {
    out.copy_(out_);
  }
}

Tensor& nan_to_num_out_mps(const Tensor& self,
                           std::optional<double> nan,
                           std::optional<double> pos_inf,
                           std::optional<double> neg_inf,
                           Tensor& result) {
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "nan_to_num: dtype of out: ",
              result.scalar_type(),
              " should be same as input: ",
              self.scalar_type());
  if (result.numel() == 0) {
    return result;
  }
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    at::native::resize_output(result, self.sizes());
    result.copy_(self);
    return result;
  }
  using namespace mps;
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* selfTensor = nil;
    MPSGraphTensor* outputTensor = nil;
    MPSGraphTensor* nanReplacementTensor = nil;
    MPSGraphTensor* posInfReplacementTensor = nil;
    MPSGraphTensor* negInfReplacementTensor = nil;
  };

  @autoreleasepool {
    std::string key = "nan_to_num" + getTensorsStringKey({self});
    MPSDataType self_dtype = getMPSScalarType(self.scalar_type());

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      newCachedGraph->nanReplacementTensor = mpsGraphRankedPlaceHolder(mpsGraph, self_dtype, @[ @1 ]);
      newCachedGraph->posInfReplacementTensor = mpsGraphRankedPlaceHolder(mpsGraph, self_dtype, @[ @1 ]);
      newCachedGraph->negInfReplacementTensor = mpsGraphRankedPlaceHolder(mpsGraph, self_dtype, @[ @1 ]);

      MPSGraphTensor* nanFreeTensor =
          [mpsGraph selectWithPredicateTensor:[mpsGraph isNaNWithTensor:newCachedGraph->selfTensor name:nil]
                          truePredicateTensor:newCachedGraph->nanReplacementTensor
                         falsePredicateTensor:newCachedGraph->selfTensor
                                         name:nil];
      MPSGraphTensor* subZeroTensor = [mpsGraph lessThanWithPrimaryTensor:nanFreeTensor
                                                          secondaryTensor:[mpsGraph constantWithScalar:0.0
                                                                                              dataType:self_dtype]
                                                                     name:nil];
      MPSGraphTensor* isInfTensor = [mpsGraph isInfiniteWithTensor:nanFreeTensor name:nil];
      // workaround for Monterey; On Ventura the output of lessThan() is always Boolean
      if (subZeroTensor.dataType != MPSDataTypeBool) {
        subZeroTensor = castMPSTensor(mpsGraph, subZeroTensor, kBool);
      }
      if (isInfTensor.dataType != MPSDataTypeBool) {
        isInfTensor = castMPSTensor(mpsGraph, isInfTensor, kBool);
      }
      MPSGraphTensor* isNegInfTensor = [mpsGraph logicalANDWithPrimaryTensor:subZeroTensor
                                                             secondaryTensor:isInfTensor
                                                                        name:nil];
      MPSGraphTensor* negInfFreeTensor = [mpsGraph selectWithPredicateTensor:isNegInfTensor
                                                         truePredicateTensor:newCachedGraph->negInfReplacementTensor
                                                        falsePredicateTensor:nanFreeTensor
                                                                        name:nil];
      newCachedGraph->outputTensor = [mpsGraph selectWithPredicateTensor:[mpsGraph isInfiniteWithTensor:negInfFreeTensor
                                                                                                   name:nil]
                                                     truePredicateTensor:newCachedGraph->posInfReplacementTensor
                                                    falsePredicateTensor:negInfFreeTensor
                                                                    name:nil];
    });
    MPSScalar nanReplacementScalar, posInfReplacementScalar, negInfReplacementScalar;
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(), "nan_to_num_mps", [&]() {
      scalar_t nan_replacement = static_cast<scalar_t>(nan.value_or(0.));
      scalar_t pos_inf_replacement =
          pos_inf.has_value() ? static_cast<scalar_t>(pos_inf.value()) : std::numeric_limits<scalar_t>::max();
      scalar_t neg_inf_replacement =
          neg_inf.has_value() ? static_cast<scalar_t>(neg_inf.value()) : std::numeric_limits<scalar_t>::lowest();

      nanReplacementScalar = getMPSScalar(nan_replacement, self.scalar_type());
      posInfReplacementScalar = getMPSScalar(pos_inf_replacement, self.scalar_type());
      negInfReplacementScalar = getMPSScalar(neg_inf_replacement, self.scalar_type());
    });

    MPSStream* stream = getCurrentMPSStream();
    Placeholder selfPlaceholder = Placeholder(cachedGraph->selfTensor, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, result);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      cachedGraph->nanReplacementTensor : getMPSGraphTensorFromScalar(stream, nanReplacementScalar),
      cachedGraph->posInfReplacementTensor : getMPSGraphTensorFromScalar(stream, posInfReplacementScalar),
      cachedGraph->negInfReplacementTensor : getMPSGraphTensorFromScalar(stream, negInfReplacementScalar),
    };
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
  return result;
}

static void isneginf_kernel_mps(TensorIteratorBase& iter) {
  mps::is_posneginf_helper(iter, true);
}

static void isposinf_kernel_mps(TensorIteratorBase& iter) {
  mps::is_posneginf_helper(iter, false);
}

REGISTER_DISPATCH(where_kernel, &where_kernel_mps)
REGISTER_DISPATCH(isneginf_stub, &isneginf_kernel_mps)
REGISTER_DISPATCH(isposinf_stub, &isposinf_kernel_mps)

} // namespace at::native
