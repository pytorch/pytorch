//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorOperators.h>
#include <ATen/mps/MPSGeneratorImpl.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/argmax.h>
#include <ATen/ops/bernoulli_native.h>
#include <ATen/ops/div.h>
#include <ATen/ops/exponential_native.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/multinomial_native.h>
#include <ATen/ops/normal_native.h>
#include <ATen/ops/random_native.h>
#include <ATen/ops/randperm.h>
#include <ATen/ops/randperm_native.h>
#include <ATen/ops/topk.h>
#include <ATen/ops/uniform_native.h>
#include <ATen/ops/view_as_real.h>
#endif

namespace at::native {
namespace mps {

struct RandomCachedGraph : public MPSCachedGraph {
  RandomCachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  // Only relevant for multinomial
  MPSGraphTensor* probTensor = nil;
  MPSGraphTensor* resultTensor = nil;
  MPSGraphTensor* stateTensor = nil;
  // used for Normal distributions only
  MPSGraphTensor *meanTensor = nil, *stdTensor = nil;
};

typedef MPSGraphTensor* (^RandomOpBlock)(RandomCachedGraph*, MPSGraphTensor*);
#define RandomOpFn(graph, randomTensor) MPSGraphTensor*(mps::RandomCachedGraph * graph, MPSGraphTensor * randomTensor)

// for Uniform distributions with scalar from (val1) and to (val2) intervals
// for Normal distributions with scalar mean (val1) and std (val2) values
template <typename scalar_t>
Tensor& random_mps_impl(Tensor& self,
                        scalar_t val1,
                        scalar_t val2,
                        const std::optional<Tensor>& mean_opt,
                        const std::optional<Tensor>& std_opt,
                        MPSGraphRandomDistribution distribution,
                        std::optional<Generator> gen,
                        std::string op_name,
                        RandomOpBlock randomBlock) {
  if (self.numel() == 0) {
    return self;
  }
  at::assert_no_internal_overlap(self);
  // MPS random is broken for 5D+ tensors, see https://github.com/pytorch/pytorch/issues/147624
  const auto need_reshape = self.ndimension() > 4;
  auto mps_gen = get_generator_or_default<MPSGeneratorImpl>(gen, at::mps::detail::getDefaultMPSGenerator());
  auto stream = getCurrentMPSStream();

  @autoreleasepool {
    auto key = op_name + getTensorsStringKey({self, mean_opt.value_or(Tensor()), std_opt.value_or(Tensor())}) + ":" +
        std::to_string(val1) + ":" + std::to_string(val2);
    auto cachedGraph = LookUpOrCreateCachedGraph<RandomCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->stateTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[ @(at::mps::detail::PHILOX_STATE_N) ]);

      // BF16, FP16, FP32 and Int32 are the only data types supported for distributions on MPS backend.
      const MPSDataType inputDataType = [&] {
        // only for random_mps, we pass interval range of type int64_t
        if constexpr (std::is_same_v<scalar_t, int64_t>) {
          return MPSDataTypeInt32;
        }
        // for bernoully always use float32
        if constexpr (std::is_same_v<scalar_t, bool>) {
          return MPSDataTypeFloat32;
        }
        switch (self.scalar_type()) {
          case kHalf:
            return MPSDataTypeFloat16;
          case kFloat:
            return MPSDataTypeFloat32;
          case kBFloat16: {
            return MPSDataTypeBFloat16;
          }
          default:
            TORCH_CHECK_TYPE(false, "Unsupported type ", self.scalar_type(), " for operation ", op_name);
        }
      }();
      const MPSDataType outputDataType = std::is_same_v<scalar_t, bool> ? MPSDataTypeBool : inputDataType;

      MPSGraphRandomOpDescriptor* desc = [MPSGraphRandomOpDescriptor descriptorWithDistribution:distribution
                                                                                       dataType:inputDataType];
      if (distribution == MPSGraphRandomDistributionUniform) {
        if (inputDataType == MPSDataTypeInt32) {
          desc.minInteger = static_cast<NSInteger>(val1);
          desc.maxInteger = static_cast<NSInteger>(val2);
        } else {
          desc.min = static_cast<float>(val1);
          desc.max = static_cast<float>(val2);
        }
      } else if (distribution == MPSGraphRandomDistributionNormal) {
        desc.mean = static_cast<float>(val1);
        desc.standardDeviation = static_cast<float>(val2);
      }
      // we don't use the output state tensor from the MPSGraph API as it requires reading back from GPU to CPU.
      // Instead, we keep the Philox state in the MPSGenerator and use the PyTorch's philox_engine to maintain
      // the counters, and feed them to the graph manually
      auto self_shape = getMPSShape(self);
      NSArray<MPSGraphTensor*>* resultTensors =
          [mpsGraph randomTensorWithShape:need_reshape ? @[ @(self.numel()) ] : self_shape
                               descriptor:desc
                              stateTensor:newCachedGraph->stateTensor
                                     name:nil];
      newCachedGraph->resultTensor =
          need_reshape ? [mpsGraph reshapeTensor:resultTensors[0] withShape:self_shape name:nil] : resultTensors[0];
      if (randomBlock) {
        newCachedGraph->resultTensor = randomBlock(newCachedGraph, newCachedGraph->resultTensor);
      }
      // results will be cast if self's scalar type isn't directly supported by MPS backend.
      if (getMPSDataType(self) != outputDataType)
        newCachedGraph->resultTensor = castMPSTensor(mpsGraph, newCachedGraph->resultTensor, self.scalar_type());
    });
    // feed the updated state values to the graph
    MPSNDArrayDescriptor* stateDesc =
        [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeInt32 shape:@[ @(at::mps::detail::PHILOX_STATE_N) ]];
    MPSNDArray* stateNDArray = [[[MPSNDArray alloc] initWithDevice:stream->device() descriptor:stateDesc] autorelease];
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(mps_gen->mutex_);
      // update the Philox state values on each run
      mps_gen->update_philox_counters();
      [stateNDArray writeBytes:mps_gen->state_data() strideBytes:nil];
    }
    MPSGraphTensorData* stateTensorData = [[[MPSGraphTensorData alloc] initWithMPSNDArray:stateNDArray] autorelease];

    Placeholder meanPlaceholder, stdPlaceholder;
    NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
    feeds[cachedGraph->stateTensor] = stateTensorData;

    if (cachedGraph->stdTensor) {
      const Tensor& stdTensor = *(at::borrow_from_optional_tensor(std_opt));
      stdPlaceholder = Placeholder(cachedGraph->stdTensor, stdTensor);
      feeds[stdPlaceholder.getMPSGraphTensor()] = stdPlaceholder.getMPSGraphTensorData();
    }
    if (cachedGraph->meanTensor) {
      const Tensor& meanTensor = *(at::borrow_from_optional_tensor(mean_opt));
      meanPlaceholder = Placeholder(cachedGraph->meanTensor, meanTensor);
      feeds[meanPlaceholder.getMPSGraphTensor()] = meanPlaceholder.getMPSGraphTensorData();
    }

    // Handle non-contiguous output tensors by creating a contiguous temporary
    const auto needs_gather = needsGather(self);
    Tensor self_ = needs_gather ? at::empty_like(self, MemoryFormat::Contiguous) : self;
    Placeholder outputPlaceholder = Placeholder(cachedGraph->resultTensor, self_);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);

    // Copy results back to original non-contiguous output
    if (needs_gather) {
      self.copy_(self_);
    }
  }

  return self;
}

static Tensor& normal_mps_impl(Tensor& self,
                               double mean_s,
                               double std_s,
                               const std::optional<Tensor>& mean_opt,
                               const std::optional<Tensor>& std_opt,
                               std::optional<Generator> gen,
                               std::string op_name) {
  const Tensor& std_t = *(at::borrow_from_optional_tensor(std_opt));
  const Tensor& mean_t = *(at::borrow_from_optional_tensor(mean_opt));

  TORCH_CHECK(std_s >= 0.0, op_name, " expects std >= 0.0, but found std ", std_s);
  if (std_t.defined()) {
    TORCH_CHECK(!std_t.is_complex(), op_name, " expects standard deviation to be non-complex");
    if (mean_t.defined())
      TORCH_CHECK(mean_t.numel() == std_t.numel(), op_name, ": mean and std must have same number of elements")
  }
  TORCH_CHECK(!(mean_t.defined() && mean_t.is_complex()), op_name, " expects mean to be non-complex");

  RandomOpBlock random_op_block = ^RandomOpFn(cachedGraph, randomTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* resultTensor = randomTensor;

    if (std_t.defined()) {
      cachedGraph->stdTensor = mpsGraphRankedPlaceHolder(mpsGraph, std_t);
      resultTensor = [mpsGraph multiplicationWithPrimaryTensor:randomTensor
                                               secondaryTensor:cachedGraph->stdTensor
                                                          name:nil];
    }
    if (mean_t.defined()) {
      cachedGraph->meanTensor = mpsGraphRankedPlaceHolder(mpsGraph, mean_t);
      return [mpsGraph additionWithPrimaryTensor:resultTensor secondaryTensor:cachedGraph->meanTensor name:nil];
    }
    return resultTensor;
  };
  if (c10::isComplexType(self.scalar_type())) {
    auto real_view = at::view_as_real(self);
    random_mps_impl<double>(real_view,
                            mean_s,
                            std_s,
                            mean_opt,
                            std_opt,
                            MPSGraphRandomDistributionNormal,
                            gen,
                            op_name + getTensorsStringKey({mean_t, std_t}),
                            random_op_block);
    return self;
  }
  return random_mps_impl<double>(self,
                                 mean_s,
                                 std_s,
                                 mean_opt,
                                 std_opt,
                                 MPSGraphRandomDistributionNormal,
                                 gen,
                                 op_name + getTensorsStringKey({mean_t, std_t}),
                                 random_op_block);
}

static Tensor& bernoulli_mps_impl(Tensor& self,
                                  const Tensor& prob_t,
                                  std::optional<Generator> gen,
                                  std::string op_name) {
  TORCH_CHECK(prob_t.is_same_size(self) || prob_t.dim() == 0,
              op_name,
              ": probability and self tensor should be of the same shape")

  RandomOpBlock random_op_block = ^RandomOpFn(cachedGraph, randomTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    cachedGraph->stdTensor = mpsGraphRankedPlaceHolder(mpsGraph, prob_t);
    return [mpsGraph lessThanWithPrimaryTensor:randomTensor
                               secondaryTensor:castMPSTensor(mpsGraph, cachedGraph->stdTensor, [randomTensor dataType])
                                          name:nil];
  };
  // Bernoulli generates binary output so we use bool type
  return mps::random_mps_impl<bool>(self,
                                    0.0,
                                    1.0,
                                    std::nullopt,
                                    prob_t,
                                    MPSGraphRandomDistributionUniform,
                                    gen,
                                    op_name + getTensorsStringKey({prob_t}),
                                    random_op_block);
}

} // namespace mps

Tensor& uniform_mps_(Tensor& self, double from, double to, std::optional<Generator> gen) {
  auto scalar_type = self.scalar_type();
  if (scalar_type == ScalarType::ComplexFloat)
    scalar_type = ScalarType::Float;
  else if (scalar_type == ScalarType::ComplexHalf)
    scalar_type = ScalarType::Half;
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, scalar_type, "check_uniform_bounds", [&] {
    const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
    const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
    TORCH_CHECK(from <= to, "uniform_ expects to return a [from, to) range, but found from=", from, " > to=", to);
    TORCH_CHECK((to - from) <= std::numeric_limits<scalar_t>::max(),
                "uniform_ expects to-from <= std::numeric_limits<",
                toString(scalar_type),
                ">::max(), but found to=",
                to,
                " and from=",
                from,
                " which result in to-from to exceed the limit");
    from = std::min(std::max(from, min), max);
    to = std::max(std::min(to, max), min);
  });

  if (c10::isComplexType(self.scalar_type())) {
    auto real_view = at::view_as_real(self);
    mps::random_mps_impl<double>(
        real_view, from, to, std::nullopt, std::nullopt, MPSGraphRandomDistributionUniform, gen, __func__, nullptr);
    return self;
  }
  return mps::random_mps_impl<double>(
      self, from, to, std::nullopt, std::nullopt, MPSGraphRandomDistributionUniform, gen, __func__, nullptr);
}

Tensor& normal_mps_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  return mps::normal_mps_impl(self, mean, std, std::nullopt, std::nullopt, gen, "normal");
}

Tensor normal_mps(const Tensor& mean, double std, std::optional<Generator> gen) {
  Tensor self = at::empty(mean.sizes(), mean.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  return mps::normal_mps_impl(self, 0.0, std, mean, std::nullopt, gen, "normal");
}

Tensor normal_mps(double mean, const Tensor& std, std::optional<Generator> gen) {
  Tensor self = at::empty(std.sizes(), std.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  // when there's no tensor-type mean, we cannot pass scalar mean value due to the order of
  // multiply/add ops in random computation. So we create a mean tensor instead.
  Tensor mean_t = at::full_like(self, Scalar(mean));
  return mps::normal_mps_impl(self, 0.0, 1.0, mean_t, std, gen, "normal");
}

Tensor normal_mps(const Tensor& mean, const Tensor& std, std::optional<Generator> gen) {
  auto shape = at::infer_size(mean.sizes(), std.sizes());
  Tensor self = at::empty(shape, mean.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  return mps::normal_mps_impl(self, 0.0, 1.0, mean, std, gen, "normal");
}

Tensor& normal_mps_out(const Tensor& mean, double std, std::optional<Generator> gen, Tensor& self) {
  return mps::normal_mps_impl(self, 0.0, std, mean, std::nullopt, gen, "normal");
}

Tensor& normal_mps_out(double mean, const Tensor& std, std::optional<Generator> gen, Tensor& self) {
  // when there's no tensor-type mean, we cannot pass scalar mean value due to the order of
  // multiply/add ops in random computation. So we create a mean tensor instead.
  Tensor mean_t = at::full_like(self, Scalar(mean));
  return mps::normal_mps_impl(self, 0.0, 1.0, mean_t, std, gen, "normal");
}

Tensor& normal_mps_out(const Tensor& mean, const Tensor& std, std::optional<Generator> gen, Tensor& self) {
  TORCH_CHECK(mean.numel() == std.numel(), "normal_mps_out: mean and std must have same number of elements")
  return mps::normal_mps_impl(self, 0.0, 1.0, mean, std, gen, "normal");
}

Tensor& bernoulli_out_mps(const Tensor& p_, std::optional<Generator> gen, Tensor& result) {
  result.resize_(p_.sizes());
  return mps::bernoulli_mps_impl(result, p_, gen, __func__);
}

Tensor& bernoulli_mps_(Tensor& self, double p, std::optional<Generator> gen) {
  TORCH_CHECK(0.0 <= p && p <= 1.0, "bernoulli_mps_ expects p to be in [0, 1], but got p=", p);
  Tensor prob_t = at::full({}, Scalar(p), c10::TensorOptions().dtype(kFloat).device(kMPS));
  return mps::bernoulli_mps_impl(self, prob_t, gen, __func__);
}

Tensor& bernoulli_mps_(Tensor& self, const Tensor& p_, std::optional<Generator> gen) {
  return mps::bernoulli_mps_impl(self, p_, gen, __func__);
}

// random_.from
Tensor& random_mps_(Tensor& self, int64_t from, std::optional<int64_t> to_opt, std::optional<Generator> gen) {
  auto input_dtype = self.scalar_type();
  int64_t to = 0;

  if (to_opt.has_value()) {
    // [from, to)
    to = *to_opt;
    TORCH_CHECK(from < to, "random_mps_ expects 'from' to be less than 'to', but got from=", from, " >= to=", to);
    if (isFloatingType(input_dtype)) {
      AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input_dtype, "random_update_from_to", [&] {
        from = templates::update_from<scalar_t>(from);
        to = templates::update_to<scalar_t>(to);
        TORCH_CHECK(from < to,
                    "random_mps_ expects 'from' casted to dtype to be less than 'to' casted to dtype, but got from=",
                    from,
                    " >= to=",
                    to);
      });
      templates::check_from_to_in_range(from, to - 1, self.dtype());
    }
  } else if (from != std::numeric_limits<int64_t>::lowest()) {
    // [from, std::numeric_limits<int64_t>::max()]
    if (isFloatingType(input_dtype)) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16, input_dtype, "random_from_to_range_calc", [&] {
            constexpr int64_t scalar_t_max = static_cast<int64_t>(1) << std::numeric_limits<scalar_t>::digits;
            to = scalar_t_max > std::numeric_limits<int64_t>::max() ? std::numeric_limits<int64_t>::max()
                                                                    : static_cast<int64_t>(scalar_t_max);
            from = templates::update_from<scalar_t>(from);
            TORCH_CHECK(
                from < to,
                "random_mps_ expects 'from' casted to dtype to be less than or equal to 'to' casted to dtype, but got from=",
                from,
                " > to=",
                to);
          });
    } else if (isIntegralType(input_dtype, /*includeBool=*/true)) {
      AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, input_dtype, "random_from_to_range_calc", [&] {
        if constexpr (std::is_same_v<scalar_t, int64_t>) {
          to = std::numeric_limits<int64_t>::max();
        } else {
          to = static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1;
        }
      });
    } else {
      TORCH_CHECK(false, "random_mps_ handles only integral, floating-point and boolean types");
    }
    templates::check_from_to_in_range(from, to - 1, self.dtype());
  } else {
    // [std::numeric_limits<int64_t>::lowest(), std::numeric_limits<int64_t>::max()]
    // range = 2^64

    // TODO - should we error out in case max is beyond MPS limit (INT32_MAX)?
    TORCH_CHECK(false, "random_mps_ currently does not handle the lowest() -> max() range");
  }

  return mps::random_mps_impl<int64_t>(
      self, from, to - 1, std::nullopt, std::nullopt, MPSGraphRandomDistributionUniform, gen, __func__, nullptr);
}

Tensor& random_mps_(Tensor& self, int64_t to, std::optional<Generator> gen) {
  return random_mps_(self, 0, to, gen);
}

Tensor& random_mps_(Tensor& self, std::optional<Generator> gen) {
  return random_mps_(self, 0, std::nullopt, gen);
}

// Exponential distribution
Tensor& exponential_mps_(Tensor& self, double lambda, std::optional<Generator> gen) {
  TORCH_CHECK(lambda > 0.0, "exponential_ expects lambda > 0.0, but found lambda=", lambda);

  mps::RandomOpBlock random_op_block = ^RandomOpFn(cachedGraph, randomTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* unitTensor = [mpsGraph constantWithScalar:1.0f dataType:randomTensor.dataType];
    MPSGraphTensor* minusLambdaTensor = [mpsGraph constantWithScalar:-lambda dataType:randomTensor.dataType];
    MPSGraphTensor* subtractTensor = [mpsGraph subtractionWithPrimaryTensor:unitTensor
                                                            secondaryTensor:randomTensor
                                                                       name:nil];
    MPSGraphTensor* logTensor = [mpsGraph logarithmWithTensor:subtractTensor name:nil];
    return [mpsGraph divisionWithPrimaryTensor:logTensor secondaryTensor:minusLambdaTensor name:nil];
  };
  auto eps = std::numeric_limits<float>::epsilon();
  return mps::random_mps_impl<double>(self,
                                      eps,
                                      1.0,
                                      std::nullopt,
                                      std::nullopt,
                                      MPSGraphRandomDistributionUniform,
                                      gen,
                                      "exponential_mps_:" + std::to_string(lambda),
                                      random_op_block);
}

Tensor& randperm_out_mps(int64_t n, std::optional<Generator> generator, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  TORCH_CHECK(!generator.has_value() || (generator.has_value() && result.device() == generator->device()),
              "Expected a '",
              result.device(),
              "' generator device but found '",
              generator->device(),
              "'");
  check_supported_max_int_with_precision(n, result);

  result.resize_({n});
  if (n == 0) {
    return result;
  }

  mps::RandomOpBlock random_op_block = ^RandomOpFn(cachedGraph, randomTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* argsortTensor = [mpsGraph argSortWithTensor:randomTensor axis:0 name:nil];
    if (result.scalar_type() != kInt) {
      argsortTensor = [mpsGraph castTensor:argsortTensor toType:mps::getMPSDataType(result) name:@"castOutput"];
    }
    return argsortTensor;
  };

  return mps::random_mps_impl<int64_t>(result,
                                       std::numeric_limits<int64_t>::min(),
                                       std::numeric_limits<int64_t>::max(),
                                       std::nullopt,
                                       std::nullopt,
                                       MPSGraphRandomDistributionUniform,
                                       generator,
                                       "ranperm_out_mps:" + mps::getTensorsStringKey({result}),
                                       random_op_block);
}

static Tensor& multinomial_with_replacement_mps_kernel(const Tensor& self,
                                                       const int64_t n_sample,
                                                       std::optional<Generator> generator,
                                                       Tensor& result) {
  using namespace mps;

  auto mps_gen = get_generator_or_default<MPSGeneratorImpl>(generator, at::mps::detail::getDefaultMPSGenerator());
  int inputSize = self.dim();
  int numDist = inputSize == 1 ? 1 : self.size(0);
  int numCategories = inputSize == 1 ? self.size(0) : self.size(1);

  // Restructure data for 2d
  auto self_v = inputSize == 1 ? self.view({numDist, numCategories}) : self;
  auto result_v = inputSize == 1 ? result.view({numDist, n_sample}) : result;

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "multinomial_with_replacement:" + getTensorsStringKey({self}) + ":" + std::to_string(n_sample);
    auto cachedGraph = LookUpOrCreateCachedGraph<RandomCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSShape* prob_shape = getMPSShape(self_v);
      newCachedGraph->stateTensor = mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[ @7 ]);

      auto prob_dtype = getMPSDataType(self_v);

      // This is probability weights
      newCachedGraph->probTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self_v), prob_shape);

      MPSGraphTensor* sumProbs = [mpsGraph reductionSumWithTensor:newCachedGraph->probTensor axis:-1 name:nil];

      MPSGraphTensor* normalizedProbs = [mpsGraph divisionWithPrimaryTensor:newCachedGraph->probTensor
                                                            secondaryTensor:sumProbs
                                                                       name:nil];

      auto ns_numCategories = [NSNumber numberWithInt:numCategories];
      auto ns_numDist = [NSNumber numberWithInt:numDist];
      auto ns_n_sample = [NSNumber numberWithInt:n_sample];

      MPSGraphTensor* ones = [mpsGraph constantWithScalar:1.0f
                                                    shape:@[ ns_numCategories, ns_numCategories ]
                                                 dataType:prob_dtype];
      auto zeroTensor = [mpsGraph constantWithScalar:0.0f dataType:MPSDataTypeInt32];
      auto minusOneTensor = [mpsGraph constantWithScalar:-1.0f dataType:MPSDataTypeInt32];

      MPSGraphTensor* upperTriangle = [mpsGraph bandPartWithTensor:ones
                                                    numLowerTensor:zeroTensor
                                                    numUpperTensor:minusOneTensor
                                                              name:nil];
      MPSGraphTensor* upperProbRange = [mpsGraph matrixMultiplicationWithPrimaryTensor:normalizedProbs
                                                                       secondaryTensor:upperTriangle
                                                                                  name:nil];

      MPSGraphTensor* lowerProbRange = [mpsGraph subtractionWithPrimaryTensor:upperProbRange
                                                              secondaryTensor:normalizedProbs
                                                                         name:nil];

      upperProbRange = [mpsGraph reshapeTensor:upperProbRange withShape:@[ ns_numDist, @1, ns_numCategories ] name:nil];
      lowerProbRange = [mpsGraph reshapeTensor:lowerProbRange withShape:@[ ns_numDist, @1, ns_numCategories ] name:nil];

      MPSGraphRandomOpDescriptor* descriptor =
          [MPSGraphRandomOpDescriptor descriptorWithDistribution:MPSGraphRandomDistributionUniform dataType:prob_dtype];
      NSArray<MPSGraphTensor*>* generatorTensors = [mpsGraph randomTensorWithShape:@[ ns_numDist, ns_n_sample, @1 ]
                                                                        descriptor:descriptor
                                                                       stateTensor:newCachedGraph->stateTensor
                                                                              name:nil];
      MPSGraphTensor* randomTensor = generatorTensors[0];

      auto broadcastShape = @[ ns_numDist, ns_n_sample, ns_numCategories ];
      int broadcastShapeVals[3] = {numDist, static_cast<int>(n_sample), numCategories};
      MPSGraphTensor* broadcastShapeTensor =
          [mpsGraph constantWithData:[NSData dataWithBytes:broadcastShapeVals length:sizeof(int) * broadcastShape.count]
                               shape:@[ [NSNumber numberWithUnsignedInteger:broadcastShape.count] ]
                            dataType:MPSDataTypeUInt32];

      MPSGraphTensor* samplesTensor = [mpsGraph broadcastTensor:randomTensor toShape:broadcastShape name:nil];
      MPSGraphTensor* sampleAbove = [mpsGraph greaterThanWithPrimaryTensor:samplesTensor
                                                           secondaryTensor:lowerProbRange
                                                                      name:nil];
      MPSGraphTensor* sampleBelow = [mpsGraph lessThanWithPrimaryTensor:samplesTensor
                                                        secondaryTensor:upperProbRange
                                                                   name:nil];
      MPSGraphTensor* sampleWithin = [mpsGraph logicalANDWithPrimaryTensor:sampleAbove
                                                           secondaryTensor:sampleBelow
                                                                      name:nil];
      MPSGraphTensor* sampleMask = [mpsGraph castTensor:sampleWithin toType:MPSDataTypeInt32 name:@"sampleMask"];
      MPSGraphTensor* categoriesTensor = [mpsGraph coordinateAlongAxis:-1
                                                       withShapeTensor:broadcastShapeTensor
                                                                  name:nil];
      MPSGraphTensor* binnedSamplesTensor = [mpsGraph multiplicationWithPrimaryTensor:categoriesTensor
                                                                      secondaryTensor:sampleMask
                                                                                 name:nil];
      MPSGraphTensor* reducedTensor = [mpsGraph reductionSumWithTensor:binnedSamplesTensor axis:-1 name:nil];
      MPSGraphTensor* reshapeTensor = [mpsGraph reshapeTensor:reducedTensor
                                                    withShape:@[ ns_numDist, ns_n_sample ]
                                                         name:nil];
      newCachedGraph->resultTensor = [mpsGraph castTensor:reshapeTensor
                                                   toType:getMPSDataType(result)
                                                     name:@"resultTensor"];
    });
    // update the Philox state values on each run of the same graph
    MPSNDArrayDescriptor* stateDesc =
        [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeInt32 shape:@[ @(at::mps::detail::PHILOX_STATE_N) ]];
    MPSNDArray* stateNDArray = [[[MPSNDArray alloc] initWithDevice:stream->device() descriptor:stateDesc] autorelease];
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(mps_gen->mutex_);
      // update the Philox state values on each run
      mps_gen->update_philox_counters();
      [stateNDArray writeBytes:mps_gen->state_data() strideBytes:nil];
    }
    MPSGraphTensorData* stateTensorData = [[[MPSGraphTensorData alloc] initWithMPSNDArray:stateNDArray] autorelease];

    auto probPlaceholder = Placeholder(cachedGraph->probTensor, self_v);
    auto outputPlaceholder = Placeholder(cachedGraph->resultTensor, result_v);
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      cachedGraph->stateTensor : stateTensorData,
      probPlaceholder.getMPSGraphTensor() : probPlaceholder.getMPSGraphTensorData()
    };
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return result;
}

/* The largest consecutive integer representable in float32 (2^24) */
constexpr int64_t FLOAT32_MAX_CONSECUTIVE_INT = 1 << (FLT_MANT_DIG);

Tensor& multinomial_out_mps(const Tensor& self,
                            int64_t n_sample,
                            bool with_replacement,
                            std::optional<Generator> gen,
                            Tensor& result) {
  TORCH_CHECK(result.device() == self.device(), "multinomial arguments must have the same device");
  TORCH_CHECK(self.dim() > 0 && self.dim() <= 2, "prob_dist must be 1 or 2 dim");
  TORCH_CHECK(at::isFloatingType(self.scalar_type()),
              "multinomial only supports floating-point dtypes for input, got: ",
              self.scalar_type());
  TORCH_CHECK(
      result.scalar_type() == ScalarType::Long, "multinomial expects Long tensor out, got: ", result.scalar_type());
  TORCH_CHECK(n_sample > 0, "cannot sample n_sample <= 0 samples");
  int64_t n_categories = self.size(-1);
  TORCH_CHECK(with_replacement || (n_sample <= n_categories),
              "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  TORCH_CHECK(n_categories <= FLOAT32_MAX_CONSECUTIVE_INT, "number of categories cannot exceed 2^24");

  if (self.dim() == 1) {
    result.resize_({n_sample});
  } else {
    const int64_t n_dist = self.size(0);
    result.resize_({n_dist, n_sample});
  }
  if (result.numel() == 0) {
    return result;
  }

  // Fast-path for no replacement (or if only one sample draw).
  // Reference:
  // https://github.com/pytorch/pytorch/issues/11931#issuecomment-625882503
  if (!with_replacement || n_sample == 1) {
    // Sanity checks on `self`.
    auto is_valid = ((self.max() < INFINITY) & (self.min() >= 0)).item();
    TORCH_CHECK(is_valid.to<bool>(), "probability tensor contains either `inf`, `nan` or element < 0");
    bool zero_prob_condition = false;
    if (self.dim() == 1) {
      zero_prob_condition = (self.sum() == 0).item().to<bool>();
    } else {
      zero_prob_condition = (self.sum(1) == 0).sum().item().to<bool>();
    }
    TORCH_CHECK(!zero_prob_condition, "invalid multinomial distribution (sum of probabilities <= 0)");

    // The algorithm is from gumbel softmax.
    // s = argmax( logp - log(-log(eps)) ) where eps ~ U(0, 1)
    // Here we can apply exp to the formula which will not affect result of
    // argmax or topk. Then we have
    // s = argmax( p / (-log(eps)) ) where eps ~ U(0, 1).
    // We can also simplify the formula above by
    // s = argmax( p / q ) where q ~ Exp(1)
    // If needed, create `q` as contiguous tensor to ensure memory layout supports inplace operations
    const auto has_strided_api = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);
    auto q = at::empty_like(self, {}, has_strided_api ? std::nullopt : std::optional(MemoryFormat::Contiguous));
    q.exponential_(1, gen);
    // In theory the probability to generate 0 from exponential distribution is
    // 0. However, on CUDA side there is a protection to avoid 0s, but on CPU
    // side, there is a very low probability to generate 0 from
    // exponential<double>. The probability is about 2^(-DBL_MANT_DIG). We just
    // ignore it here, but there may be some risk to get invalid output on CPU.
    at::div_out(q, self, q);
    if (n_sample == 1) {
      at::argmax_out(result, q, /*dim=*/-1, /*keepdim=*/true);
    } else {
      Tensor vals = at::empty(result.sizes(), self.options());
      at::topk_out(vals, result, q, n_sample);
    }
    return result;
  }

  result = multinomial_with_replacement_mps_kernel(const_cast<Tensor&>(self), n_sample, gen, result);

  return result;
}

Tensor multinomial_mps(const Tensor& self, int64_t n_sample, bool with_replacement, std::optional<Generator> gen) {
  Tensor result = at::empty({0}, self.options().dtype(kLong));
  multinomial_out_mps(self, n_sample, with_replacement, gen, result);
  return result;
}

} // namespace at::native
