//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/bincount_native.h>
namespace at::native {

static Tensor& bincount_mps_impl(const Tensor& self, const Tensor& weights, Tensor& output) {
  using namespace mps;

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* weightsTensor_ = nil;
    MPSGraphTensor* scatterDataTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();
  bool has_weights = weights.defined();

  @autoreleasepool {
    string key = "bincount_mps_impl" + getTensorsStringKey({self, weights});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* scatterDataTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSScalarType(output.scalar_type()));

      MPSGraphTensor* updatesTensor = nil;
      if (has_weights) {
        updatesTensor = mpsGraphRankedPlaceHolder(mpsGraph, weights);
      } else {
        updatesTensor = [mpsGraph constantWithScalar:1.0f shape:getMPSShape(self) dataType:getMPSDataType(output)];
      }

      MPSGraphTensor* castedInputTensor = inputTensor;
      if (self.scalar_type() == kByte) {
        castedInputTensor = [mpsGraph castTensor:inputTensor toType:MPSDataTypeInt32 name:@"castInputTensor"];
      }

      MPSGraphTensor* outputTensor = [mpsGraph scatterWithDataTensor:scatterDataTensor
                                                       updatesTensor:updatesTensor
                                                       indicesTensor:castedInputTensor
                                                                axis:0
                                                                mode:MPSGraphScatterModeAdd
                                                                name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
      newCachedGraph->scatterDataTensor_ = scatterDataTensor;
      if (has_weights) {
        newCachedGraph->weightsTensor_ = updatesTensor;
      }
    });

    // Create placeholders which use the keys of the CachedGraph to create inputs and outputs of the operation
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    Placeholder scatterPlaceholder = Placeholder(cachedGraph->scatterDataTensor_, output);
    Placeholder weightsPlaceholder = Placeholder();

    // Create dictionary of inputs/feeds and outputs/results
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary dictionary];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[scatterPlaceholder.getMPSGraphTensor()] = scatterPlaceholder.getMPSGraphTensorData();
    if (has_weights) {
      weightsPlaceholder = Placeholder(cachedGraph->weightsTensor_, weights);
      feeds[weightsPlaceholder.getMPSGraphTensor()] = weightsPlaceholder.getMPSGraphTensorData();
    }

    // Run the graph
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return output;
}

Tensor _bincount_mps(const Tensor& self, const c10::optional<Tensor>& weights_opt, int64_t minlength) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weights_maybe_owned = at::borrow_from_optional_tensor(weights_opt);
  const Tensor& weights = *weights_maybe_owned;

  TORCH_CHECK(c10::isIntegralType(self.scalar_type(), /*includesBool=*/true));
  TORCH_CHECK(minlength >= 0, "minlength should be >= 0");

  if (self.dim() == 1 && self.numel() == 0) {
    return at::zeros({minlength}, kLong, c10::nullopt /* layout */, kMPS, c10::nullopt /* pin_memory */);
  }
  TORCH_CHECK(self.dim() == 1 && self.min().item<int64_t>() >= 0,
              "bincount only supports 1-d non-negative integral inputs.");

  bool has_weights = weights.defined();
  TORCH_CHECK(!(has_weights && (weights.dim() != 1 || weights.size(0) != self.size(0))),
              "weights should be 1-d and have the same length as input");

  const int64_t nbins = std::max(self.max().item<int64_t>() + 1L, minlength);
  Tensor output;

  Tensor weights_ = weights;
  if (has_weights) {
    if (weights.scalar_type() != ScalarType::Float && weights.scalar_type() != ScalarType::Int &&
        weights.scalar_type() != ScalarType::Half) {
      // Scatter doesn't work for int8/int16 dtypes
      weights_ = weights.to(kInt);
    }
    output = at::zeros({nbins},
                       optTypeMetaToScalarType(weights_.options().dtype_opt()),
                       weights_.options().layout_opt(),
                       weights_.options().device_opt(),
                       weights_.options().pinned_memory_opt());
  } else {
    output = at::zeros({nbins}, kLong, c10::nullopt /* layout */, kMPS, c10::nullopt /* pin_memory */);
  }

  return bincount_mps_impl(self, weights_, output);
}

} // namespace at::native
