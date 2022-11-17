//  Copyright © 2022 Apple Inc.

#include <ATen/native/Distributions.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/core/PhiloxRNGEngine.h>

namespace at {
namespace native {
namespace mps {

struct RandomCachedGraph : public MPSCachedGraph
{
  RandomCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {
    // initialize Philox state values (only required once when graph is created)
    const auto seed = c10::detail::getNonDeterministicRandom();
    const auto subsequence = c10::detail::getNonDeterministicRandom();
    philoxState = at::Philox4_32(seed, subsequence);
    // the two last state values are the Philox keys which are initialized once only
    stateValues[5] = static_cast<uint32_t>(seed);
    stateValues[6] = static_cast<uint32_t>(seed >> 32);
  }
  // Only relevant for multinomial
  MPSGraphTensor *probTensor = nil;
  MPSGraphTensor *resultTensor = nil;
  MPSGraphTensor *stateTensor = nil;
  // used for Normal distributions only
  MPSGraphTensor *meanTensor = nil, *stdTensor = nil;
  // we initialize and keep the philox's state in the graph. This would
  // guarantee producing new random values each time the same graph is reused.
  at::Philox4_32 philoxState;
  std::array<uint32_t, 7> stateValues = {1};

  void updatePhiloxCounters() {
    // calling philoxState() would call operator() of philox_engine class to
    // get each of the four newly generated counter values (see PhiloxRNGEngine.h).
    for (int i = 1; i <= 4; i++)
      stateValues[i] = philoxState();
  }
};

typedef MPSGraphTensor* (^RandomOpBlock)(RandomCachedGraph*, MPSGraphTensor*);
#define RandomOpFn(graph, randomTensor) MPSGraphTensor* (mps::RandomCachedGraph* graph, MPSGraphTensor* randomTensor)

// for Uniform distributions with scalar from (val1) and to (val2) intervals
// for Normal distributions with scalar mean (val1) and std (val2) values
template<typename scalar_t>
Tensor& random_mps_impl(Tensor& self, scalar_t val1, scalar_t val2,
                        const c10::optional<Tensor>& mean_opt,
                        const c10::optional<Tensor>& std_opt,
                        MPSGraphRandomDistribution distribution,
                        std::string op_name, RandomOpBlock randomBlock)
{
  if (self.numel() == 0) {
    return self;
  }
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = op_name + getTensorsStringKey({self}) + ":" + to_string(val1) + ":" + to_string(val2);
    auto cachedGraph = cache_->LookUpAs<RandomCachedGraph>(key);

    if (!cachedGraph) {
      cachedGraph = cache_->CreateCachedGraphAs<RandomCachedGraph>(key, ^ MPSCachedGraph * () {
        RandomCachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new RandomCachedGraph(mpsGraph);
          newCachedGraph->stateTensor = mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[@7]);

          // FP16, FP32 and Int32 are the only data types supported for distributions on MPS backend.
          const MPSDataType inputDataType = [&] {
            // only for random_mps, we pass interval range of type int64_t
            if (std::is_same<scalar_t, int64_t>::value)
              return MPSDataTypeInt32;
            else
              return (self.scalar_type() == ScalarType::Half) ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
          }();
          const MPSDataType outputDataType = (std::is_same<scalar_t, bool>::value) ? MPSDataTypeBool : inputDataType;

          MPSGraphRandomOpDescriptor *desc = [MPSGraphRandomOpDescriptor descriptorWithDistribution: distribution
                                                                                           dataType: inputDataType];
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
          // Instead, we keep the Philox state in the cached graph and use the PyTorch's philox_engine to maintain
          // the counters, and feed them to the graph manually
          NSArray<MPSGraphTensor*> *resultTensors = [mpsGraph randomTensorWithShape: getMPSShape(self)
                                                                         descriptor: desc
                                                                        stateTensor: newCachedGraph->stateTensor
                                                                               name: nil];
          newCachedGraph->resultTensor = randomBlock ? randomBlock(newCachedGraph, resultTensors[0]) : resultTensors[0];
          // results will be cast if self's scalar type isn't directly supported by MPS backend.
          if (getMPSDataType(self.scalar_type()) != outputDataType)
            newCachedGraph->resultTensor = castMPSTensor(mpsGraph, newCachedGraph->resultTensor, self.scalar_type());
        }
        return newCachedGraph;
      });
    }
    // update the Philox state values on each run of the same graph
    cachedGraph->updatePhiloxCounters();
    // feed the updated state values to the graph
    MPSNDArrayDescriptor *stateDesc = [MPSNDArrayDescriptor descriptorWithDataType: MPSDataTypeInt32 shape: @[@7]];
    MPSNDArray *stateNDArray = [[[MPSNDArray alloc] initWithDevice: stream->device() descriptor: stateDesc] autorelease];
    [stateNDArray writeBytes: &cachedGraph->stateValues[0] strideBytes: nil];
    MPSGraphTensorData* stateTensorData = [[[MPSGraphTensorData alloc] initWithMPSNDArray: stateNDArray] autorelease];

    Placeholder meanPlaceholder, stdPlaceholder;
    NSMutableDictionary *feeds = [[NSMutableDictionary new] autorelease];
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

    Placeholder outputPlaceholder = Placeholder(cachedGraph->resultTensor, self);
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData(),
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return self;
}

Tensor& normal_mps_impl(Tensor& self, double mean_s, double std_s,
                        const c10::optional<Tensor>& mean_opt,
                        const c10::optional<Tensor>& std_opt,
                        std::string op_name)
{
  const Tensor& std_t  = *(at::borrow_from_optional_tensor(std_opt));
  const Tensor& mean_t = *(at::borrow_from_optional_tensor(mean_opt));

  TORCH_CHECK(std_s >= 0.0, op_name, " expects std >= 0.0, but found std=", std_s);
  if (std_t.defined()) {
    TORCH_CHECK(!std_t.is_complex(), op_name, " expects standard deviation to be non-complex");
    if (mean_t.defined())
      TORCH_CHECK(mean_t.numel() == std_t.numel(), op_name, ": mean and std must have same number of elements")
  }

  RandomOpBlock random_op_block = ^RandomOpFn(cachedGraph, randomTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* resultTensor = randomTensor;

    if (std_t.defined()) {
      cachedGraph->stdTensor = mpsGraphRankedPlaceHolder(mpsGraph, std_t);
      resultTensor = [mpsGraph multiplicationWithPrimaryTensor: randomTensor
                                               secondaryTensor: cachedGraph->stdTensor
                                                          name: nil];
    }
    if (mean_t.defined()) {
      cachedGraph->meanTensor = mpsGraphRankedPlaceHolder(mpsGraph, mean_t);
      return [mpsGraph additionWithPrimaryTensor: resultTensor
                                 secondaryTensor: cachedGraph->meanTensor
                                            name: nil];
    }
    return resultTensor;
  };
  return random_mps_impl<double>(self, mean_s, std_s, mean_opt, std_opt,
                                 MPSGraphRandomDistributionNormal,
                                 op_name + getTensorsStringKey({mean_t, std_t}), random_op_block);

}

Tensor& bernoulli_mps_impl(Tensor& self, const Tensor& prob_t, std::string op_name)
{
  TORCH_CHECK(prob_t.is_same_size(self), op_name, ": probability and self tensor should be of the same shape")

  RandomOpBlock random_op_block = ^RandomOpFn(cachedGraph, randomTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    cachedGraph->stdTensor = mpsGraphRankedPlaceHolder(mpsGraph, prob_t);
    return [mpsGraph lessThanWithPrimaryTensor: randomTensor
                               secondaryTensor: cachedGraph->stdTensor
                                          name: nil];
  };
  // Bernoulli generates binary output so we use bool type
  return mps::random_mps_impl<bool>(self, 0.0, 1.0, c10::nullopt, prob_t,
                                    MPSGraphRandomDistributionUniform,
                                    op_name + getTensorsStringKey({prob_t}), random_op_block);
}

} // namespace mps

Tensor& uniform_mps_(Tensor& self, double from, double to, c10::optional<Generator> gen) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "check_uniform_bounds", [&] {
    const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
    const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
    TORCH_CHECK(from <= to, "uniform_ expects to return a [from, to) range, but found from=", from, " > to=", to);
    TORCH_CHECK((to - from) <= std::numeric_limits<scalar_t>::max(),
          "uniform_ expects to-from <= std::numeric_limits<", toString(self.scalar_type()),
          ">::max(), but found to=", to, " and from=", from,
          " which result in to-from to exceed the limit");
    from = std::min(std::max(from, min), max);
    to = std::max(std::min(to, max), min);
  });

  return mps::random_mps_impl<double>(self, from, to, c10::nullopt, c10::nullopt,
                                      MPSGraphRandomDistributionUniform, __func__, nullptr);
}

Tensor& normal_mps_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  return mps::normal_mps_impl(self, mean, std, c10::nullopt, c10::nullopt, __func__);
}

Tensor normal_mps(const Tensor& mean, double std, c10::optional<Generator> gen) {
  Tensor self = empty_mps(mean.sizes(), mean.scalar_type(), c10::nullopt, kMPS);
  return mps::normal_mps_impl(self, 0.0, std, mean, c10::nullopt, __func__);
}

Tensor normal_mps(double mean, const Tensor& std, c10::optional<Generator> gen) {
  Tensor self = empty_mps(std.sizes(), std.scalar_type(), c10::nullopt, kMPS);
  // when there's no tensor-type mean, we cannot pass scalar mean value due to the order of
  // multiply/add ops in random computation. So we create a mean tensor instead.
  Tensor mean_t = at::full_like(self, Scalar(mean));
  return mps::normal_mps_impl(self, 0.0, 1.0, mean_t, std, __func__);
}

Tensor normal_mps(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen) {
  auto shape = at::infer_size(mean.sizes(), std.sizes());
  Tensor self = empty_mps(shape, mean.scalar_type(), c10::nullopt, kMPS);
  return mps::normal_mps_impl(self, 0.0, 1.0, mean, std, __func__);
}

Tensor& normal_mps_out(const Tensor& mean, double std, c10::optional<Generator> gen, Tensor& self) {
  return mps::normal_mps_impl(self, 0.0, std, mean, c10::nullopt, __func__);
}

Tensor& normal_mps_out(double mean, const Tensor& std, c10::optional<Generator> gen, Tensor& self) {
  // when there's no tensor-type mean, we cannot pass scalar mean value due to the order of
  // multiply/add ops in random computation. So we create a mean tensor instead.
  Tensor mean_t = at::full_like(self, Scalar(mean));
  return mps::normal_mps_impl(self, 0.0, 1.0, mean_t, std, __func__);
}

Tensor& normal_mps_out(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen, Tensor& self) {
  TORCH_CHECK(mean.numel() == std.numel(), "normal_mps_out: mean and std must have same number of elements")
  return mps::normal_mps_impl(self, 0.0, 1.0, mean, std, __func__);
}

Tensor& bernoulli_out_mps(const Tensor& p_, c10::optional<Generator> gen, Tensor& result) {
  result.resize_(p_.sizes());
  return  mps::bernoulli_mps_impl(result, p_, __func__);
}

Tensor& bernoulli_mps_(Tensor& self, double p, c10::optional<Generator> gen) {
  TORCH_CHECK(0.0 <= p && p <= 1.0, "bernoulli_mps_ expects p to be in [0, 1], but got p=", p);
  Tensor prob_t = at::full_like(self, Scalar(p));
  return mps::bernoulli_mps_impl(self, prob_t, __func__);
}

Tensor& bernoulli_mps_(Tensor& self, const Tensor& p_, c10::optional<Generator> gen) {
  return mps::bernoulli_mps_impl(self, p_, __func__);
}

// random_.from
Tensor& random_mps_(Tensor& self, int64_t from, optional<int64_t> to_opt, c10::optional<Generator> gen) {
  auto input_dtype = self.scalar_type();
  int64_t to = 0;

  if (to_opt.has_value()) {
    // [from, to)
    to = *to_opt;
    TORCH_CHECK(from < to, "random_mps_ expects 'from' to be less than 'to', but got from=", from, " >= to=", to);
    if (isFloatingType(input_dtype)) {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_dtype, "random_update_from_to", [&] {
        from = templates::update_from<scalar_t>(from);
        to = templates::update_to<scalar_t>(to);
        TORCH_CHECK(from < to, "random_mps_ expects 'from' casted to dtype to be less than 'to' casted to dtype, but got from=", from, " >= to=", to);
      });
      templates::check_from_to_in_range(from, to - 1, self.dtype());
    }
  } else if (from != std::numeric_limits<int64_t>::lowest()) {
    // [from, std::numeric_limits<int64_t>::max()]
    if (isFloatingType(input_dtype)) {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_dtype, "random_from_to_range_calc", [&] {
        constexpr int64_t scalar_t_max = static_cast<int64_t>(1) << std::numeric_limits<scalar_t>::digits;
        to = scalar_t_max > std::numeric_limits<int64_t>::max() ? std::numeric_limits<int64_t>::max() : static_cast<int64_t>(scalar_t_max);
        from = templates::update_from<scalar_t>(from);
        TORCH_CHECK(from < to, "random_mps_ expects 'from' casted to dtype to be less than or equal to 'to' casted to dtype, but got from=", from, " > to=", to);
      });
    } else if (isIntegralType(input_dtype, /*includeBool=*/true)) {
      AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, input_dtype, "random_from_to_range_calc", [&] {
        if (std::is_same<scalar_t, bool>::value) {
          to = static_cast<int64_t>(true);
        } else {
          to = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
        }
      });
    }
    else {
      TORCH_CHECK(false, "random_mps_ handles only integral, floating-point and boolean types");
    }
    templates::check_from_to_in_range(from, to, self.dtype());
  }
  else {
    // [std::numeric_limits<int64_t>::lowest(), std::numeric_limits<int64_t>::max()]
    // range = 2^64

    // TODO - should we error out in case max is beyond MPS limit (INT32_MAX)?
    TORCH_CHECK(false, "random_mps_ currently does not handle the lowest() -> max() range");
  }

  return mps::random_mps_impl<int64_t>(self, from, to - 1, c10::nullopt, c10::nullopt,
                                       MPSGraphRandomDistributionUniform, __func__, nullptr);
}

Tensor& random_mps_(Tensor& self, int64_t to, c10::optional<Generator> gen) {
  return random_mps_(self, 0, to, gen);
}

// Exponential distribution
Tensor& exponential_mps_(Tensor& self, double lambda, c10::optional<Generator> gen) {
  TORCH_CHECK(lambda > 0, "exponential_mps_: lambda must be greater than zero")

  mps::RandomOpBlock random_op_block = ^RandomOpFn(cachedGraph, randomTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* unitTensor = [mpsGraph constantWithScalar: 1.0f
                                                     dataType: randomTensor.dataType];
    MPSGraphTensor* minusLambdaTensor = [mpsGraph constantWithScalar: -lambda
                                                            dataType: randomTensor.dataType];
    MPSGraphTensor* subtractTensor = [mpsGraph subtractionWithPrimaryTensor: unitTensor
                                                            secondaryTensor: randomTensor
                                                                       name: nil];
    MPSGraphTensor* logTensor = [mpsGraph logarithmWithTensor: subtractTensor
                                                         name: nil];
    return [mpsGraph divisionWithPrimaryTensor: logTensor
                               secondaryTensor: minusLambdaTensor
                                          name: nil];
  };
  return mps::random_mps_impl<double>(self, 0.0, 1.0, c10::nullopt, c10::nullopt,
                                      MPSGraphRandomDistributionUniform,
                                      "exponential_mps_:" + std::to_string(lambda), random_op_block);
}

Tensor& multinomial_with_replacement_mps_kernel(
    const Tensor& self,
    const int64_t n_sample,
    c10::optional<Generator> generator,
    Tensor& result) {

  using namespace mps;

  int inputSize = self.dim();
  int numDist =
      inputSize == 1 ? 1 : self.size(0);
  int numCategories =
      inputSize == 1 ? self.size(0) : self.size(1);

  // Restructure data for 2d
  auto self_v = inputSize == 1 ? self.view({numDist, numCategories}) : self;
  auto result_v = inputSize == 1 ? result.view({numDist, n_sample}) : result;

  MPSStream* stream = getCurrentMPSStream();
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = "multinomial_with_replacement:" + getTensorsStringKey({self}) + ":" + to_string(n_sample);
    auto cachedGraph = cache_->LookUpAs<RandomCachedGraph>(key);
    if (!cachedGraph) {
      cachedGraph = cache_->CreateCachedGraphAs<RandomCachedGraph>(key, ^ MPSCachedGraph * () {
        RandomCachedGraph *newCachedGraph = nil;
        @autoreleasepool {
          MPSShape* prob_shape = getMPSShape(self_v);
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new RandomCachedGraph(mpsGraph);
          newCachedGraph->stateTensor = mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[@7]);

          auto prob_dtype = getMPSDataType(self_v.scalar_type());

          // This is probability weights
          newCachedGraph->probTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self_v.scalar_type()), prob_shape);

          MPSGraphTensor *sumProbs = [mpsGraph reductionSumWithTensor:newCachedGraph->probTensor
                                                                 axis:-1
                                                                 name:nil];

          MPSGraphTensor *normalizedProbs = [mpsGraph divisionWithPrimaryTensor:newCachedGraph->probTensor
                                                                secondaryTensor:sumProbs
                                                                           name:nil];

          auto ns_numCategories = [NSNumber numberWithInt:numCategories];
          auto ns_numDist = [NSNumber numberWithInt:numDist];
          auto ns_n_sample = [NSNumber numberWithInt:n_sample];

          MPSGraphTensor *ones = [mpsGraph constantWithScalar:1.0f
                                                        shape:@[ns_numCategories, ns_numCategories]
                                                     dataType:prob_dtype];
          auto zeroTensor = [mpsGraph constantWithScalar: 0.0f
                                                dataType: MPSDataTypeInt32];
          auto minusOneTensor = [mpsGraph constantWithScalar: -1.0f
                                                    dataType: MPSDataTypeInt32];

          MPSGraphTensor *upperTriangle = [mpsGraph bandPartWithTensor:ones
                                                        numLowerTensor:zeroTensor
                                                        numUpperTensor:minusOneTensor
                                                                  name:nil];
          MPSGraphTensor *upperProbRange = [mpsGraph matrixMultiplicationWithPrimaryTensor:normalizedProbs
                                                                           secondaryTensor:upperTriangle
                                                                                      name:nil];

          MPSGraphTensor *lowerProbRange = [mpsGraph subtractionWithPrimaryTensor:upperProbRange
                                                                  secondaryTensor:normalizedProbs
                                                                             name:nil];

          upperProbRange = [mpsGraph reshapeTensor:upperProbRange
                                         withShape:@[ns_numDist, @1, ns_numCategories]
                                              name:nil];
          lowerProbRange = [mpsGraph reshapeTensor:lowerProbRange
                                         withShape:@[ns_numDist, @1, ns_numCategories]
                                              name:nil];

          MPSGraphRandomOpDescriptor *descriptor = [MPSGraphRandomOpDescriptor descriptorWithDistribution:MPSGraphRandomDistributionUniform
                                                                                                 dataType:prob_dtype];
          NSArray<MPSGraphTensor*> *generatorTensors = [mpsGraph randomTensorWithShape:@[ns_numDist, ns_n_sample, @1]
                                                                            descriptor:descriptor
                                                                           stateTensor:newCachedGraph->stateTensor
                                                                                  name:nil];
          MPSGraphTensor *randomTensor = generatorTensors[0];

          auto broadcastShape = @[ns_numDist ,ns_n_sample, ns_numCategories];
          int broadcastShapeVals[3] = {numDist, n_sample, numCategories};
          MPSGraphTensor *broadcastShapeTensor = [mpsGraph constantWithData:[NSData dataWithBytes:broadcastShapeVals length:sizeof(int) * broadcastShape.count]
                                                                      shape:@[[NSNumber numberWithUnsignedInteger:broadcastShape.count]]
                                                                   dataType:MPSDataTypeUInt32];

          MPSGraphTensor *samplesTensor = [mpsGraph broadcastTensor:randomTensor
                                                            toShape:broadcastShape
                                                               name:nil];
          MPSGraphTensor *sampleAbove = [mpsGraph greaterThanWithPrimaryTensor:samplesTensor
                                                               secondaryTensor:lowerProbRange
                                                                          name:nil];
          MPSGraphTensor *sampleBelow = [mpsGraph lessThanWithPrimaryTensor:samplesTensor
                                                            secondaryTensor:upperProbRange
                                                                       name:nil];
          MPSGraphTensor *sampleWithin = [mpsGraph logicalANDWithPrimaryTensor:sampleAbove
                                                            secondaryTensor:sampleBelow
                                                                       name:nil];
          MPSGraphTensor *sampleMask = [mpsGraph castTensor:sampleWithin
                                                     toType:MPSDataTypeInt32
                                                       name:@"sampleMask"];
          MPSGraphTensor *categoriesTensor = [mpsGraph coordinateAlongAxis:-1
                                                           withShapeTensor:broadcastShapeTensor
                                                                      name:nil];
          MPSGraphTensor *binnedSamplesTensor = [mpsGraph multiplicationWithPrimaryTensor:categoriesTensor
                                                                       secondaryTensor:sampleMask
                                                                                  name:nil];
          MPSGraphTensor *reducedTensor = [mpsGraph reductionSumWithTensor:binnedSamplesTensor
                                                                      axis:-1
                                                                      name:nil];
          MPSGraphTensor *reshapeTensor = [mpsGraph reshapeTensor:reducedTensor
                                                       withShape:@[ns_numDist ,ns_n_sample]
                                                            name:nil];
          newCachedGraph->resultTensor = [mpsGraph castTensor:reshapeTensor
                                                       toType:getMPSDataType(result.scalar_type())
                                                         name:@"resultTensor"];
        }
        return newCachedGraph;
     });
    }
    // update the Philox state values on each run of the same graph
    cachedGraph->updatePhiloxCounters();
    // feed the updated state values to the graph
    MPSNDArrayDescriptor *stateDesc = [MPSNDArrayDescriptor descriptorWithDataType: MPSDataTypeInt32 shape: @[@7]];
    MPSNDArray *stateNDArray = [[[MPSNDArray alloc] initWithDevice: stream->device() descriptor: stateDesc] autorelease];
    [stateNDArray writeBytes: &cachedGraph->stateValues[0] strideBytes: nil];
    MPSGraphTensorData* stateTensorData = [[[MPSGraphTensorData alloc] initWithMPSNDArray: stateNDArray] autorelease];

    auto probPlaceholder = Placeholder(cachedGraph->probTensor, self_v);
    auto outputPlaceholder = Placeholder(cachedGraph->resultTensor, result_v);
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      cachedGraph->stateTensor : stateTensorData,
      probPlaceholder.getMPSGraphTensor() : probPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return result;

}

/* The largest consecutive integer representable in float32 (2^24) */
constexpr int64_t FLOAT32_MAX_CONSECUTIVE_INT = 1 << (FLT_MANT_DIG);

Tensor& multinomial_out_mps(const Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    c10::optional<Generator> gen,
    Tensor& result) {

  TORCH_CHECK(
      result.device() == self.device(),
      "multinomial arguments must have the same device");
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "prob_dist must be 1 or 2 dim");
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()),
      "multinomial only supports floating-point dtypes for input, got: ",
      self.scalar_type());
  TORCH_CHECK(result.scalar_type() == ScalarType::Long,
      "multinomial expects Long tensor out, got: ", result.scalar_type());
  TORCH_CHECK(n_sample > 0, "cannot sample n_sample <= 0 samples");
  int64_t n_categories = self.size(-1);
  TORCH_CHECK(with_replacement || (n_sample <= n_categories),
      "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  TORCH_CHECK(
      n_categories <= FLOAT32_MAX_CONSECUTIVE_INT,
      "number of categories cannot exceed 2^24");

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
    TORCH_CHECK(
        is_valid.to<bool>(),
        "probability tensor contains either `inf`, `nan` or element < 0");
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    bool zero_prob_condition;
    if (self.dim() == 1){
      zero_prob_condition = (self.sum() == 0).item().to<bool>();
    } else {
      zero_prob_condition = (self.sum(1) == 0).sum().item().to<bool>();
    }
    TORCH_CHECK(
        !zero_prob_condition,
        "invalid multinomial distribution (sum of probabilities <= 0)");

    // The algorithm is from gumbel softmax.
    // s = argmax( logp - log(-log(eps)) ) where eps ~ U(0, 1)
    // Here we can apply exp to the formula which will not affect result of
    // argmax or topk. Then we have
    // s = argmax( p / (-log(eps)) ) where eps ~ U(0, 1).
    // We can also simplify the formula above by
    // s = argmax( p / q ) where q ~ Exp(1)
    Tensor q = at::empty_like(self).exponential_(1, gen);
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

Tensor multinomial_mps(
    const Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    c10::optional<Generator> gen) {
  Tensor result = at::empty({0}, self.options().dtype(kLong));
  multinomial_out_mps(self, n_sample, with_replacement, gen, result);
  return result;
}

} // namespace native
} // namespace at
