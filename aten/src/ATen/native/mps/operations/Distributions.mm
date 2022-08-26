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

// for Uniform distributions with scalar from and to intervals
// for Normal distributions with scalar or tensor mean and std values
Tensor& random_mps_impl(Tensor& self, double val1, double val2,
                        const c10::optional<Tensor>& mean_opt,
                        const c10::optional<Tensor>& std_opt,
                        MPSGraphRandomDistribution distribution, std::string op_name)
{
  if (self.numel() == 0) {
    return self;
  }
  const Tensor& meanTensor = *(at::borrow_from_optional_tensor(mean_opt));
  const Tensor& stdTensor = *(at::borrow_from_optional_tensor(std_opt));
  const bool hasMeanTensor = meanTensor.defined();
  const bool hasStdTensor = stdTensor.defined();

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = op_name + getTensorsStringKey({self, meanTensor, stdTensor}) +
                 (!hasMeanTensor ? (":" + to_string(val1)) : "") +
                 (!hasStdTensor  ? (":" + to_string(val2)) : "");
    RandomCachedGraph* cachedGraph = static_cast<RandomCachedGraph *>(cache_->LookUp(key));

    if (!cachedGraph) {
      cachedGraph = static_cast<RandomCachedGraph *>(cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        RandomCachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new RandomCachedGraph(mpsGraph);
          newCachedGraph->stateTensor = mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[@7]);

          MPSGraphRandomOpDescriptor *desc = [MPSGraphRandomOpDescriptor descriptorWithDistribution: distribution
                                                                                           dataType: getMPSScalarType(self.scalar_type())];
          if (distribution == MPSGraphRandomDistributionUniform) {
            desc.min = static_cast<float>(val1);
            desc.max = static_cast<float>(val2);
          } else if (distribution == MPSGraphRandomDistributionNormal) {
            if (!hasMeanTensor)
              desc.mean = static_cast<float>(val1);
            if (!hasStdTensor)
              desc.standardDeviation = static_cast<float>(val2);
          }
          // we don't use the output state tensor from the MPSGraph API as it requires reading back from GPU to CPU.
          // Instead, we keep the Philox state in the cached graph and use the PyTorch's philox_engine to maintain
          // the counters, and feed them to the graph manually
          NSArray<MPSGraphTensor*> *resultTensors = [mpsGraph randomTensorWithShape: getMPSShape(self)
                                                                         descriptor: desc
                                                                        stateTensor: newCachedGraph->stateTensor
                                                                               name: nil];
          newCachedGraph->resultTensor = resultTensors[0];
          // these would run only for Normal distributions where mean and std tensors are defined.
          MPSGraphTensor* scaleTensor = newCachedGraph->resultTensor;
          if (hasStdTensor) {
            newCachedGraph->stdTensor = mpsGraphRankedPlaceHolder(mpsGraph, stdTensor);
            scaleTensor = [mpsGraph multiplicationWithPrimaryTensor: resultTensors[0]
                                                    secondaryTensor: newCachedGraph->stdTensor
                                                               name: nil];
            newCachedGraph->resultTensor = scaleTensor;
          }
          if (hasMeanTensor) {
            newCachedGraph->meanTensor = mpsGraphRankedPlaceHolder(mpsGraph, meanTensor);
            newCachedGraph->resultTensor = [mpsGraph additionWithPrimaryTensor: scaleTensor
                                                               secondaryTensor: newCachedGraph->meanTensor
                                                                          name: nil];
          }
        }
        return newCachedGraph;
      }));
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
    if (hasStdTensor) {
      stdPlaceholder = Placeholder(cachedGraph->stdTensor, stdTensor);
      feeds[stdPlaceholder.getMPSGraphTensor()] = stdPlaceholder.getMPSGraphTensorData();
    }
    if (hasMeanTensor) {
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

  return mps::random_mps_impl(self, from, to, c10::nullopt, c10::nullopt, MPSGraphRandomDistributionUniform, __func__);
}

Tensor& normal_mps_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  TORCH_CHECK(std >= 0.0, "normal_mps_ expects std >= 0.0, but found std=", std);
  return mps::random_mps_impl(self, mean, std, c10::nullopt, c10::nullopt, MPSGraphRandomDistributionNormal, __func__);
}

Tensor normal_mps(const Tensor& mean, double std, c10::optional<Generator> gen) {
  TORCH_CHECK(std >= 0.0, "normal_mps_ expects std >= 0.0, but found std=", std);
  Tensor self = empty_mps(mean.sizes(), mean.scalar_type(), c10::nullopt, kMPS);
  return mps::random_mps_impl(self, 0.0, std, mean, c10::nullopt, MPSGraphRandomDistributionNormal, __func__);
}

Tensor normal_mps(double mean, const Tensor& std, c10::optional<Generator> gen) {
  Tensor self = empty_mps(std.sizes(), std.scalar_type(), c10::nullopt, kMPS);
  // when there's no tensor-type mean, we cannot pass scalar mean value due to the order of
  // multiply/add ops in random computation. So we create a mean tensor instead.
  Tensor mean_t = at::full_like(self, Scalar(mean));
  return mps::random_mps_impl(self, 0.0, 1.0, mean_t, std, MPSGraphRandomDistributionNormal, __func__);
}

Tensor normal_mps(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen) {
  TORCH_CHECK(!std.is_complex(), "normal expects standard deviation to be non-complex");
  TORCH_CHECK(mean.numel() == std.numel(), "normal_mps_out: mean and std must have same number of elements")
  auto shape = at::infer_size(mean.sizes(), std.sizes());
  Tensor self = empty_mps(shape, mean.scalar_type(), c10::nullopt, kMPS);
  return mps::random_mps_impl(self, 0.0, 1.0, mean, std, MPSGraphRandomDistributionNormal, __func__);
}

Tensor& normal_mps_out(const Tensor& mean, double std, c10::optional<Generator> gen, Tensor& self) {
  TORCH_CHECK(std >= 0.0, "normal_mps_out expects std >= 0.0, but found std=", std);
  return mps::random_mps_impl(self, 0.0, std, mean, c10::nullopt, MPSGraphRandomDistributionNormal, __func__);
}

Tensor& normal_mps_out(double mean, const Tensor& std, c10::optional<Generator> gen, Tensor& self) {
  TORCH_CHECK(!std.is_complex(), "normal expects standard deviation to be non-complex");
  // when there's no tensor-type mean, we cannot pass scalar mean value due to the order of
  // multiply/add ops in random computation. So we create a mean tensor instead.
  Tensor mean_t = at::full_like(self, Scalar(mean));
  return mps::random_mps_impl(self, 0.0, 1.0, mean_t, std, MPSGraphRandomDistributionNormal, __func__);
}

Tensor& normal_mps_out(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen, Tensor& self) {
  TORCH_CHECK(!std.is_complex(), "normal expects standard deviation to be non-complex");
  TORCH_CHECK(mean.numel() == std.numel(), "normal_mps_out: mean and std must have same number of elements")
  return mps::random_mps_impl(self, 0.0, 1.0, mean, std, MPSGraphRandomDistributionNormal, __func__);
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

} // namespace native
} // namespace at
