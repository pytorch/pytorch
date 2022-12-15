//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>

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

void set_apparent_shapes(NSArray<NSNumber*> * input_shape,
                         NSArray<NSNumber*> * &apparent_input_shape,
                         int64_t num_input_dims,
                         IntArrayRef repeats,
                         NSMutableArray<NSNumber*> * &repeats_shape,
                         int64_t num_repeat_dims) {


  bool repeat_empty = false;
  if(num_repeat_dims == 0) {
    num_repeat_dims = num_input_dims;
    repeat_empty = true;
  }

  // Set repeats_shape
  repeats_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_repeat_dims];

  for(int i = 0; i < num_repeat_dims; i++) {
    if(repeat_empty)
      repeats_shape[i] = [NSNumber numberWithInteger:1];
    else
      repeats_shape[i] = [NSNumber numberWithInteger:repeats[i]];
  }

  // If no extension of the shape is needed
  if(num_repeat_dims == num_input_dims) {
    apparent_input_shape = input_shape;
  }
  // num_repeat_dims > num_input_dims
  else {
    auto rc = [NSMutableArray<NSNumber*> arrayWithCapacity:num_repeat_dims];

    for(int i = 0; i < num_repeat_dims - num_input_dims; i++)
      rc[i] = @1;

    for(int i = num_repeat_dims - num_input_dims; i < num_repeat_dims; i++)
      rc[i] = input_shape[i + num_input_dims - num_repeat_dims];
    apparent_input_shape = rc;
  }

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

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  NSArray<NSNumber*> *apparent_input_shape = nil;
  NSMutableArray<NSNumber*> *repeats_shape = nil;

  auto input_shape = getMPSShape(self);
  auto num_input_dims = [input_shape count];
  auto num_repeat_dims = repeats.size();

  set_apparent_shapes(input_shape,
                      apparent_input_shape,
                      num_input_dims,
                      repeats,
                      repeats_shape,
                      num_repeat_dims);

  // Set output shape
  std::vector<int64_t> output_shape(num_repeat_dims);
  bool zero_tensor = false;
  for(auto i : c10::irange(num_repeat_dims)) {
    output_shape[i] = repeats[i] * [apparent_input_shape[i] intValue];
    if(output_shape[i] == 0) {
      zero_tensor = true;
    }
  }

  Tensor output = at::native::empty_mps(
                      IntArrayRef(output_shape),
                      self.scalar_type(),
                      c10::nullopt,
                      kMPS,
                      c10::nullopt,
                      c10::nullopt);

  // Empty output
  if(zero_tensor || output.numel() == 0)
    return output;

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {

    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];
    NSString* ns_repeats_key = [[repeats_shape valueForKey:@"description"] componentsJoinedByString:@","];

    string key = "repeat_mps:" + getMPSTypeString(self.scalar_type())
                               + ":" + string([ns_shape_key UTF8String])
                               + ":" + string([ns_repeats_key UTF8String]);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self.scalar_type()), apparent_input_shape);
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

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, apparent_input_shape);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return output;

}

}
}
