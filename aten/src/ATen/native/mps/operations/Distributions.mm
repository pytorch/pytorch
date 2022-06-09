//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/NativeFunctions.h>
#include <ATen/AccumulateType.h>
#include <torch/library.h>
namespace at {
namespace native {
namespace templates {

}

Tensor& uniform_mps_(Tensor& input, double from, double to, c10::optional<Generator> gen_)
{
  using namespace mps;

  if (input.numel() == 0) {
    return input;
  }
  double delta = (to - from);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "check_uniform_bounds", [&] {
    const auto dtype = input.dtype();
    const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
    const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
    TORCH_CHECK(from <= to, "uniform_ expects to return a [from, to) range, but found from=", from, " > to=", to);
    TORCH_CHECK((to - from) <= std::numeric_limits<scalar_t>::max(),
          "uniform_ expects to-from <= std::numeric_limits<", toString(input.scalar_type()),
          ">::max(), but found to=", to, " and from=", from,
          " which result in to-from to exceed the limit");
    from = std::min(std::max(from, min), max);
    to = std::max(std::min(to, max), min);
  });

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  MPSStream* stream = getCurrentMPSStream();
  uint64_t seed_ = c10::detail::getNonDeterministicRandom(true);

  @autoreleasepool {
    MPSShape* input_shape = getMPSShape(input);
    string key = "uniform_mps_" + getTensorsStringKey(input) + ":" + to_string(from) + ":" + to_string(to) + ":" + to_string(seed_);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          // TODO: right now taking the default seed. Extend it to be extracted from the
          // MPSGenerator
          MPSGraphTensor* randomTensor = [mpsGraph randomUniformTensorWithShape:input_shape
                                                                           seed:seed_
                                                                           name:nil];
          MPSGraphTensor* deltaTensor = [mpsGraph constantWithScalar:delta
                                                               shape:input_shape
                                                            dataType:MPSDataTypeFloat32];
          MPSGraphTensor* fromTensor = [mpsGraph constantWithScalar:from
                                                              shape:input_shape
                                                           dataType:MPSDataTypeFloat32];
          MPSGraphTensor* mulTensor = [mpsGraph multiplicationWithPrimaryTensor:randomTensor
                                                                secondaryTensor:deltaTensor
                                                                           name:nil];
          MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:mulTensor
                                                             secondaryTensor:fromTensor
                                                                        name:nil];
          newCachedGraph->outputTensor_ = outputTensor;

        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, input);
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = nil;
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);

  }

  return input;
}

Tensor& normal_mps_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  if (self.numel() == 0)
    return self;
  TORCH_CHECK(std >= 0.0, "normal_mps_ expects std >= 0.0, but found std=", std);

  Tensor mean_t = empty_mps(
                      self.sizes(),
                      self.scalar_type(),
                      c10::nullopt,
                      kMPS,
                      c10::nullopt,
                      c10::nullopt);
  mean_t.fill_(mean);

  Tensor std_t = empty_mps(
                      self.sizes(),
                      self.scalar_type(),
                      c10::nullopt,
                      kMPS,
                      c10::nullopt,
                      c10::nullopt);
  std_t.fill_(std);

  return normal_mps_out(mean_t, std_t, gen, self);
}

Tensor& normal_mps_out(const Tensor& mean, double std, c10::optional<Generator> gen, Tensor& output) {
  TORCH_CHECK(std >= 0.0, "normal_mps_out expects std >= 0.0, but found std=", std);

  Tensor std_t = empty_mps(
                      output.sizes(),
                      output.scalar_type(),
                      c10::nullopt,
                      kMPS,
                      c10::nullopt,
                      c10::nullopt);
  std_t.fill_(std);

  return normal_mps_out(mean, std_t, gen, output);

}

Tensor& normal_mps_out(double mean, const Tensor& std, c10::optional<Generator> gen, Tensor& output) {
  TORCH_CHECK(
    std.min().ge(0).item<bool>(),
    "normal expects all elements of std >= 0.0");


  Tensor mean_t = empty_mps(
                      output.sizes(),
                      output.scalar_type(),
                      c10::nullopt,
                      kMPS,
                      c10::nullopt,
                      c10::nullopt);
  mean_t.fill_(mean);

  return normal_mps_out(mean_t, std, gen, output);

}

Tensor& normal_mps_out(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen, Tensor& output) {
  TORCH_CHECK(!std.is_complex(), "normal expects standard deviation to be non-complex");
  TORCH_CHECK(std.numel() == 0 || std.min().ge(0).item<bool>(), "normal expects all elements of std >= 0.0");
  // Check that mean and std have same number of elements
  TORCH_CHECK(mean.numel() == std.numel(), "normal_mps_out: mean and std must have same number of elements")

  using namespace mps;

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* outputTensor_ = nil;
    MPSGraphTensor* meanTensor_ = nil;
    MPSGraphTensor* stdTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  MPSStream* stream = getCurrentMPSStream();
  uint64_t seed_ = c10::detail::getNonDeterministicRandom(true);

  @autoreleasepool {
    MPSShape* input_shape = getMPSShape(output);
    string key = "normal_mps_out:" + getMPSShapeString(input_shape) + ":" + getMPSTypeString(output.scalar_type()) + ":" + to_string(seed_);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphRandomOpDescriptor* desc = [[MPSGraphRandomOpDescriptor new] autorelease];
          desc.distribution = MPSGraphRandomDistributionNormal;
          desc.dataType = getMPSDataType(output.scalar_type());
          desc.mean = 0.0;
          desc.standardDeviation = 1.0;

          MPSGraphTensor* meanTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(output.scalar_type()), input_shape);
          MPSGraphTensor* stdTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(output.scalar_type()), input_shape);

          // TODO: right now taking the default seed. Extend it to be extracted from the
          // MPSGenerator
          MPSGraphTensor* randomTensor = [mpsGraph randomTensorWithShape:input_shape
                                                              descriptor:desc
                                                                    seed:seed_
                                                                    name:nil];
          MPSGraphTensor* scaleTensor = [mpsGraph multiplicationWithPrimaryTensor:randomTensor
                                                                  secondaryTensor:stdTensor
                                                                             name:nil];
          MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:scaleTensor
                                                            secondaryTensor:meanTensor
                                                                        name:nil];
          newCachedGraph->meanTensor_ = meanTensor;
          newCachedGraph->stdTensor_ = stdTensor;
          newCachedGraph->outputTensor_ = outputTensor;

        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    auto meanPlaceholder = Placeholder(cachedGraph->meanTensor_, mean);
    auto stdPlaceholder = Placeholder(cachedGraph->stdTensor_, std);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      meanPlaceholder.getMPSGraphTensor() : meanPlaceholder.getMPSGraphTensorData(),
      stdPlaceholder.getMPSGraphTensor() : stdPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);

  }

  return output;
}

Tensor& bernoulli_out_mps(const Tensor& p_, c10::optional<Generator> gen, Tensor& result) {
  result.resize_(p_.sizes());
  return  bernoulli_mps_(result, p_, gen);
}

Tensor& bernoulli_mps_(Tensor& self, double p, c10::optional<Generator> gen) {
  TORCH_CHECK(0 <= p && p <= 1, "bernoulli_mps_ expects p to be in [0, 1], but got p=", p);
  Tensor p_t = empty_mps(
                      self.sizes(),
                      self.scalar_type(),
                      c10::nullopt,
                      kMPS,
                      c10::nullopt,
                      c10::nullopt);
  p_t.fill_(p);

  return bernoulli_mps_(self, p_t, gen);
}

Tensor& bernoulli_mps_(Tensor& self, const Tensor& p_, c10::optional<Generator> gen) {
  TORCH_CHECK(self.is_same_size(p_), "bernoulli_mps_: probability and self tensor should be of the same shape")

  using namespace mps;

  MPSStream* stream = getCurrentMPSStream();
  uint64_t seed_ = c10::detail::getNonDeterministicRandom(true);

  @autoreleasepool {
    MPSShape* input_shape = getMPSShape(self);

    auto mps_dtype = getMPSDataType(p_.scalar_type());

    MPSGraph* mpsGraph = make_mps_graph();

    MPSGraphTensor* probTensor = mpsGraphRankedPlaceHolder(mpsGraph, mps_dtype, input_shape);

    // TODO: right now taking the default seed. Extend it to be extracted from the
    // MPSGenerator
    MPSGraphTensor* randomTensor = [mpsGraph randomUniformTensorWithShape:input_shape
                                                                     seed:seed_
                                                                     name:nil];
    MPSGraphTensor* outputTensor = [mpsGraph lessThanWithPrimaryTensor:randomTensor
                                                       secondaryTensor:probTensor
                                                                  name:nil];

    auto probPlaceholder = Placeholder(probTensor, p_);
    auto outputPlaceholder = Placeholder(outputTensor, self);
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      probPlaceholder.getMPSGraphTensor() : probPlaceholder.getMPSGraphTensorData(),
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, mpsGraph, feeds, results);
  }

  return self;

}

// Taken from ATen/native/DistributionTemplates.h
#define CHECK_OUT_OF_BOUNDS(var, name, min, max, dtype) \
  TORCH_CHECK(var >= min && var <= max, name , " is out of bounds for ", dtype); \

#define WARN_OUT_OF_BOUNDS(var, name, digits, dtype) \
  if (var < -(1LL << digits) || var > (1LL << digits)) { \
    TORCH_WARN(name , " is out of bounds [-(2^", digits, "), 2^", digits, "]. ", \
      "Due to precision limitations ", dtype, " can support discrete uniform distribution only within this range. ", \
      "This warning will become an error in version 1.7 release, please fix the code in advance"); \
  }

// Modified from ATen/native/DistributionTemplates.h
static void check_from_to_in_range(int64_t from, int64_t to_inc, ScalarType scalar_type) {
  const auto dtype = scalarTypeToTypeMeta(scalar_type);
  if (isFloatingType(scalar_type)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "check_random_fp_bounds", [&] {
      const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
      const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
      CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
      CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", min, max, dtype);

      constexpr auto digits = std::numeric_limits<scalar_t>::digits;
      WARN_OUT_OF_BOUNDS(from, "from", digits, dtype);
      WARN_OUT_OF_BOUNDS(to_inc, "to - 1", digits, dtype);
    });
  } else if (isIntegralType(scalar_type, /*includeBool=*/true)) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, scalar_type, "check_random_integral_bounds", [&]() {
      const auto min = static_cast<int64_t>(std::numeric_limits<scalar_t>::lowest());
      const auto max = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
      CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
      CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", min, max, dtype);
    });
  } else {
    TORCH_CHECK(false, "check_random_bounds handles only integral, floating-point and boolean types");
  }
}


// random_.from
Tensor& random_mps_
  (Tensor& self,
   int64_t from,
   optional<int64_t> to_opt,
   c10::optional<Generator> gen) {

  using namespace mps;

  MPSStream* stream = getCurrentMPSStream();
  uint64_t seed_ = c10::detail::getNonDeterministicRandom(true);

  auto input_dtype = self.scalar_type();

  int64_t to;

  if(to_opt.has_value()) {
    // [from, to)
    to = *to_opt;
    TORCH_CHECK(from < to, "random_mps_ expects 'from' to be less than 'to', but got from=", from, " >= to=", to);
    if (isFloatingType(input_dtype)) {
      // TODO: what is "random_update_from_to"?
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_dtype, "random_update_from_to", [&] {
        from = templates::update_from<scalar_t>(from);
        to = templates::update_to<scalar_t>(to);
        TORCH_CHECK(from < to, "random_mps_ expects 'from' casted to dtype to be less than 'to' casted to dtype, but got from=", from, " >= to=", to);
      });
      check_from_to_in_range(from, to - 1, input_dtype);
    }
  }
  else if (from != std::numeric_limits<int64_t>::lowest()) {
    // [from, std::numeric_limits<int64_t>::max()]
    to = 0;
    if(isFloatingType(input_dtype)) {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_dtype, "random_from_to_range_calc", [&] {
        constexpr int64_t scalar_t_max = static_cast<int64_t>(1) << std::numeric_limits<scalar_t>::digits;
        to = scalar_t_max > std::numeric_limits<int64_t>::max() ? std::numeric_limits<int64_t>::max() : static_cast<int64_t>(scalar_t_max);
        from = templates::update_from<scalar_t>(from);
        TORCH_CHECK(from < to, "random_mps_ expects 'from' casted to dtype to be less than or equal to 'to' casted to dtype, but got from=", from, " > to=", to);
      });
    }
    else if(isIntegralType(input_dtype, /*includeBool=*/true)) {
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
    check_from_to_in_range(from, to, input_dtype);
  }
  else {
    // [std::numeric_limits<int64_t>::lowest(), std::numeric_limits<int64_t>::max()]
    // range = 2^64

    // TODO - how to implement this?
    TORCH_CHECK(false, "random_mps_ currently does not handle the lowest() -> max() range");

  }

  @autoreleasepool {
    MPSShape* input_shape = getMPSShape(self);

    MPSGraph* mpsGraph = make_mps_graph();

    MPSGraphRandomOpDescriptor* descriptor = [MPSGraphRandomOpDescriptor descriptorWithDistribution:MPSGraphRandomDistributionUniform
                                                                                           dataType:MPSDataTypeInt32];
    descriptor.minInteger = from;
    descriptor.maxInteger = to - 1;

    // TODO: right now taking the default seed. Extend it to be extracted from the
    // MPSGenerator
    MPSGraphTensor* randomTensor = [mpsGraph randomTensorWithShape:input_shape
                                                        descriptor:descriptor
                                                              seed:seed_
                                                              name:nil];

    MPSGraphTensor* outputTensor = nil;

    if(input_dtype != ScalarType::Int)
      outputTensor = [mpsGraph castTensor:randomTensor
                                   toType:getMPSDataType(input_dtype)
                                     name:@"outputTensor"];
    else
      outputTensor = randomTensor;

    auto outputPlaceholder = Placeholder(outputTensor, self);
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = nil;
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, mpsGraph, feeds, results);
  }

  return self;

}

Tensor& random_mps_
  (Tensor& self,
   int64_t to,
   c10::optional<Generator> gen) {

  return random_mps_(self, 0, to, gen);
}

// Exponential distribution

Tensor& exponential_mps_(Tensor& self, double lambda, c10::optional<Generator> gen) {

  using namespace mps;

  if (self.numel() == 0) {
    return self;
  }

  TORCH_CHECK(lambda > 0, "exponential_mps_: lambda must be greater than zero")

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  MPSStream* stream = getCurrentMPSStream();
  uint64_t seed_ = c10::detail::getNonDeterministicRandom(true);

  @autoreleasepool {
    MPSShape* self_shape = getMPSShape(self);

    MPSGraph* mpsGraph = make_mps_graph();
    // TODO: right now taking the default seed. Extend it to be extracted from the
    // MPSGenerator
    MPSGraphTensor* randomTensor = [mpsGraph randomUniformTensorWithShape:self_shape
                                                                     seed:seed_
                                                                     name:nil];
    MPSGraphTensor* unitTensor = [mpsGraph constantWithScalar:1.0f
                                                     dataType:MPSDataTypeFloat32];
    MPSGraphTensor* minusLambdaTensor = [mpsGraph constantWithScalar:-lambda
                                                       dataType:MPSDataTypeFloat32];
    MPSGraphTensor* subtractTensor = [mpsGraph subtractionWithPrimaryTensor:unitTensor
                                                            secondaryTensor:randomTensor
                                                                       name:nil];
    MPSGraphTensor* logTensor = [mpsGraph logarithmWithTensor:subtractTensor
                                                         name:nil];
    MPSGraphTensor* outputTensor = [mpsGraph divisionWithPrimaryTensor:logTensor
                                                       secondaryTensor:minusLambdaTensor
                                                                  name:nil];

    if(getMPSDataType(self.scalar_type()) != MPSDataTypeFloat32)
      outputTensor = [mpsGraph castTensor:outputTensor
                                   toType:getMPSDataType(self.scalar_type())
                                     name:@"output"];

    auto outputPlaceholder = Placeholder(outputTensor, self);
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = nil;
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, mpsGraph, feeds, results);

  }

  return self;

}

// Multinomial with replacement
Tensor multinomial_with_replacement_mps_entry(
    const Tensor& self,
    const int64_t n_sample,
    c10::optional<Generator> generator) {

  Tensor result = at::empty({0}, self.options().dtype(kLong));

  if (self.dim() == 1) {
    result.resize_({n_sample});
  } else {
    const int64_t n_dist = self.size(0);
    result.resize_({n_dist, n_sample});
  }

  return multinomial_with_replacement_mps_kernel(self, n_sample, generator, result);

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
  uint64_t seed_ = c10::detail::getNonDeterministicRandom(true);

  @autoreleasepool {
    MPSShape* prob_shape = getMPSShape(self_v);
    MPSGraph* mpsGraph = make_mps_graph();

    auto prob_dtype = getMPSDataType(self_v.scalar_type());
    auto result_dtype = getMPSDataType(result.scalar_type());

    // This is probability weights
    MPSGraphTensor *probTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self_v.scalar_type()), prob_shape);

    MPSGraphTensor *sumProbs = [mpsGraph reductionSumWithTensor:probTensor
                                                           axis:-1
                                                           name:nil];

    MPSGraphTensor *normalizedProbs = [mpsGraph divisionWithPrimaryTensor:probTensor
                                                          secondaryTensor:sumProbs
                                                                     name:nil];

    auto ns_numCategories = [NSNumber numberWithInt:numCategories];
    auto ns_numDist = [NSNumber numberWithInt:numDist];
    auto ns_n_sample = [NSNumber numberWithInt:n_sample];

    MPSGraphTensor *ones = [mpsGraph constantWithScalar:1.0f
                                                  shape:@[ns_numCategories, ns_numCategories]
                                               dataType:prob_dtype];
    MPSGraphTensor *upperTriangle = [mpsGraph bandPartWithTensor:ones
                                                        numLower:0
                                                        numUpper:-1
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

    MPSGraphTensor *stateTensor = [mpsGraph randomPhiloxStateTensorWithSeed:seed_
                                                                       name:nil];
    MPSGraphRandomOpDescriptor *descriptor = [MPSGraphRandomOpDescriptor descriptorWithDistribution:MPSGraphRandomDistributionUniform
                                                                                           dataType:prob_dtype];
    NSArray<MPSGraphTensor*> *generatorTensors = [mpsGraph randomTensorWithShape:@[ns_numDist, ns_n_sample, @1]
                                                                      descriptor:descriptor
                                                                     stateTensor:stateTensor
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
    MPSGraphTensor *resultTensor = [mpsGraph castTensor:reshapeTensor
                                                 toType:getMPSDataType(result.scalar_type())
                                                   name:@"resultTensor"];

    auto probPlaceholder = Placeholder(probTensor, self_v);
    auto outputPlaceholder = Placeholder(resultTensor, result_v);
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      probPlaceholder.getMPSGraphTensor() : probPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, mpsGraph, feeds, results);
  }

  // TODO: If I change the view, does the original get altered? I should think so
  return result;

}

} // namespace native
} // namespace at
