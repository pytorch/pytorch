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
    RandomCachedGraph* cachedGraph = static_cast<RandomCachedGraph *>(cache_->LookUp(key));

    if (!cachedGraph) {
      cachedGraph = static_cast<RandomCachedGraph *>(cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
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

} // namespace native
} // namespace at
