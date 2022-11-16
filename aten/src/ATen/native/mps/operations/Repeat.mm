//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Repeat.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>
#include <c10/util/irange.h>

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif


namespace at {
namespace native {

Tensor permute_mps(const Tensor& self, IntArrayRef dims) {
  auto nDims = self.dim();
  TORCH_CHECK(dims.size() == (size_t)nDims,
           "number of dims don't match in permute");
  auto oldSizes = self.sizes();
  auto oldStrides = self.strides();
  DimVector newSizes(nDims);
  DimVector newStrides(nDims);
  std::vector<bool> seen(nDims);
  for (const auto i : c10::irange(nDims)) {
    auto dim = maybe_wrap_dim(dims[i], nDims);
    TORCH_CHECK(!seen[dim],
             "repeated dim in permute");
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    newStrides[i] = oldStrides[dim];
  }
  return self.as_strided(newSizes, newStrides);
}

Tensor repeat_mps(const Tensor& self, IntArrayRef repeats) {

  using namespace mps;

  TORCH_CHECK(repeats.size() >= (size_t)self.dim(),
           "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  // Add new leading dimensions to the tensor if the
  // number of target dimensions is larger than the
  // number of source dimensions.
  int64_t num_new_dimensions = repeats.size() - self.dim();
  DimVector padded_size(num_new_dimensions, 1);
  padded_size.insert(padded_size.end(), self.sizes().begin(), self.sizes().end());
  DimVector target_size(repeats.size());
  bool zero_tensor = false;
  for(const auto idx : c10::irange(repeats.size())) {
    if (repeats[idx] == 0) {
      zero_tensor = true;
    }
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  Tensor expanded_tensor = self.expand(padded_size);
  Tensor result = at::empty(target_size, self.options());
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  if(zero_tensor || result.numel() == 0) {
    return result;
  }

  auto stream = at::mps::getCurrentMPSStream();
  auto inputDataType = getMPSDataType(expanded_tensor.scalar_type());
  auto outputDataType = getMPSDataType(result.scalar_type());
  if (!is_macos_13_or_newer()) {
     if (expanded_tensor.scalar_type() == kBool) {
      inputDataType = MPSDataTypeInt8;
     }
     if (result.scalar_type() == kBool) {
      outputDataType = MPSDataTypeInt8;
     }
  }

  @autoreleasepool {
    string key = "repeat_mps:" + getTensorsStringKey(self) + ":" + getArrayRefString(repeats);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, inputDataType, getMPSShape(expanded_tensor));
          MPSGraphTensor* outputTensor = [mpsGraph tileTensor:inputTensor
                                               withMultiplier:getMPSShape(repeats)
                                                         name:nil];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(
      cachedGraph->inputTensor_, expanded_tensor, /*mpsShape=*/nil, /*gatherTensorData=*/true, inputDataType);
    Placeholder outputPlaceholder = Placeholder(
      cachedGraph->outputTensor_, result, /*mpsShape=*/nil, /*gatherTensorData*/false, outputDataType);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return result;
}

Tensor repeat_interleave_mps(const Tensor& repeats,c10::optional<int64_t> output_size) {

  Tensor output;
  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();

 struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  NSNumber *repeats_shape;

  if (output_size.has_value()) {
    repeats_shape = [NSNumber numberWithInteger:output_size.value()];
  } else {
    Tensor cumsum = repeats.cumsum(0);
    repeats_shape = [NSNumber numberWithInteger:cumsum[-1].item<int64_t>()];
    TORCH_CHECK(
        (repeats >= 0).all().item<uint8_t>(), "repeats can not be negative");
  }


  @autoreleasepool {
    // A key is used to identify the MPSGraph which was created once, and can be reused if the parameters, data types etc match the earlier created MPSGraph
    string key = "repeat_interleave_mps:" + getTensorsStringKey({repeats})+ ":" + string([repeats_shape UTF8String]);

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);


          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(repeats.scalar_type()), getMPSShape(repeats));
          MPSGraphTensor* outputTensor = [mpsGraph tileTensor:inputTensor
                                               withMultiplier:repeats_shape
                                                         name:nil];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });

      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, repeats, getMPSShape(repeats));
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* output = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, output);
  }

  return output;
}

}
}
