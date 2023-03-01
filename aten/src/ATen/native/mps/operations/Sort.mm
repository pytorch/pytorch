//  Copyright © 2023 Apple Inc.

#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>

namespace at::native {

// sort
TORCH_IMPL_FUNC(sort_stable_out_mps)
(const Tensor& self,
 c10::optional<bool> stable,
 int64_t dim,
 bool descending,
 const Tensor& values,
 const Tensor& indices) {
  using namespace mps;
  values.copy_(self);
  // check if self is scalar
  dim = maybe_wrap_dim(dim, self.dim(), true);
  if (self.dim() == 0 && self.numel() == 1) {
    indices.zero_();
    return;
  }

  if (!is_macos_13_or_newer()) {
    TORCH_WARN_ONCE("torch.sort is supported by MPS on MacOS 13+, please upgrade. Falling back to CPU");
    Tensor cpu_indices = indices.clone().to("cpu");
    Tensor cpu_values = values.clone().to("cpu");
    at::sort_out(cpu_values, cpu_indices, self.to(at::Device(kCPU)), false, dim, descending);
    values.copy_(cpu_values);
    indices.copy_(cpu_indices);
    return;
  }
  if (self.scalar_type() == ScalarType::Long) {
    TORCH_WARN_ONCE("MPS: no support for int64 min/max ops, casting it to int32");
  }

  MPSStream* stream = getCurrentMPSStream();
  struct CachedGraph : public MPSCachedGraph {
      CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
      MPSGraphTensor *selfTensor = nil, *valuesTensor = nil, *indicesTensor = nil;
  };
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  @autoreleasepool {
    // Input as placeholders
    MPSShape* input_shape = getMPSShape(self);
    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];
    string key = string("sort:") + [ns_shape_key UTF8String] + ":" + getMPSTypeString(self.scalar_type()) +
                           ":dim" + to_string(dim) + ":descending" + to_string(descending);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
        cachedGraph = static_cast<CachedGraph*>(cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;
        @autoreleasepool {
            MPSGraph* mpsGraph = make_mps_graph();
            newCachedGraph = new CachedGraph(mpsGraph);
            newCachedGraph->selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self.scalar_type()), input_shape);

            MPSGraphTensor* castInputTensor = castToIHFTypes(mpsGraph, newCachedGraph->selfTensor, self);
            MPSGraphTensor * sortedTensor = [mpsGraph sortWithTensor:castInputTensor
                                                                axis:(NSInteger)dim
                                                                descending:(BOOL)descending
                                                                name:@"sort_out"];
            sortedTensor = castFromIHFTypes(mpsGraph, sortedTensor, values);
            MPSGraphTensor* argSortedTensor = [mpsGraph argSortWithTensor:castInputTensor
                                                                     axis:(NSInteger)dim
                                                                     descending:(BOOL)descending
                                                                     name:@"argsort_out"];
            argSortedTensor = castFromIHFTypes(mpsGraph, argSortedTensor, indices);
            newCachedGraph->valuesTensor = sortedTensor;
            newCachedGraph->indicesTensor = argSortedTensor;
        }
        return newCachedGraph;
      }));
    }
    Placeholder inputPlaceholder  = Placeholder(cachedGraph->selfTensor, self);
    // Outputs as placeholders
    Placeholder valuesPlaceholder = Placeholder(cachedGraph->valuesTensor, values);
    Placeholder indicesPlaceholder = Placeholder(cachedGraph->indicesTensor, indices);
    // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =  nil;
    feeds = @{ inputPlaceholder.getMPSGraphTensor() :
        inputPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
    valuesPlaceholder.getMPSGraphTensor() :
            valuesPlaceholder.getMPSGraphTensorData(),
    indicesPlaceholder.getMPSGraphTensor() :
          indicesPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}
}
