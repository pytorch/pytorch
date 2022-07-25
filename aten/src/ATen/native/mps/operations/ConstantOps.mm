//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/OperationUtils.h>

namespace at {
namespace native {

Tensor& fill_scalar_mps_impl(Tensor& self, const Scalar& value) {
  using namespace mps;

  if (self.numel() == 0) {
    return self;
  }

  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache *cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = "fill_scalar_mps_impl" + getTensorsStringKey(self) + ":" + to_string(value.toDouble());

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {

      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool{
          MPSGraph *mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = [mpsGraph constantWithScalar:value.toDouble()
                                                               shape:getMPSShape(self)
                                                            dataType:getMPSScalarType(self.scalar_type())];
          MPSGraphTensor* outputTensor = [mpsGraph identityWithTensor:inputTensor
                                                                 name:nil];

          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, self);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = nil;

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return self;
}

// returns false if tensor cannot be filled with fillBuffer()
bool fill_mps_tensor_(Tensor& self, uint8_t value) {
  if (self.is_contiguous()) {
    MPSStream* stream = getCurrentMPSStream();
    auto storage_byte_offset = self.storage_offset() * self.itemsize();
    stream->fill(mps::getMTLBufferStorage(self), 0, self.nbytes(), storage_byte_offset);
    return true;
  }
  return false;
}

Tensor& zero_mps_(Tensor& self) {
  // check if it's possible to use fillBuffer() to fill the Tensor's storage
  if (fill_mps_tensor_(self, 0) == true)
    return self;
  return fill_scalar_mps_impl(self, 0.0f);
}

Tensor& fill_scalar_mps(Tensor& self, const Scalar& value) {
  if (value.toDouble() == 0.0 && fill_mps_tensor_(self, 0) == true)
    return self;
  return fill_scalar_mps_impl(self, value);
}

Tensor& fill_tensor_mps_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  Scalar scalar_value = value.item();
  if (scalar_value.toDouble() == 0.0 && fill_mps_tensor_(self, 0) == true)
    return self;
  return fill_scalar_mps_impl(self, scalar_value);
}

} // namespace native
} // namespace at
