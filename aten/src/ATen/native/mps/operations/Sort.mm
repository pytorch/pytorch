//  Copyright © 2023 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/sort.h>
#include <ATen/ops/sort_native.h>
#endif
namespace at::native {

// sort
TORCH_IMPL_FUNC(sort_stable_out_mps)
(const Tensor& self,
 std::optional<bool> stable,
 int64_t dim,
 bool descending,
 const Tensor& values,
 const Tensor& indices) {
  using namespace mps;

  bool macOS13_3_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
  MPS_CHECK_INT64_OP_SUPPORTED(self, macOS13_3_plus, "sort_stable_out");

  if (self.numel() == 0) {
    return;
  }

  values.copy_(self);
  // check if self is scalar
  dim = maybe_wrap_dim(dim, self.dim(), true);
  if (self.dim() == 0 && self.numel() == 1) {
    indices.zero_();
    return;
  }

  MPSStream* stream = getCurrentMPSStream();
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *selfTensor = nil, *valuesTensor = nil, *indicesTensor = nil;
  };
  @autoreleasepool {
    // Input as placeholders
    MPSShape* input_shape = getMPSShape(self);
    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];
    string key = string("sort:") + [ns_shape_key UTF8String] + ":" + getMPSTypeString(self) + ":dim" +
        std::to_string(dim) + ":descending" + std::to_string(descending);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), input_shape);

      MPSGraphTensor* castInputTensor =
          castToIHFTypes(mpsGraph, newCachedGraph->selfTensor, self, /*includesInt64=*/macOS13_3_plus);
      MPSGraphTensor* sortedTensor = [mpsGraph sortWithTensor:castInputTensor
                                                         axis:(NSInteger)dim
                                                   descending:(BOOL)descending
                                                         name:@"sort_out"];
      if ([sortedTensor dataType] != getMPSDataType(values)) {
        sortedTensor = castMPSTensor(mpsGraph, sortedTensor, values.scalar_type());
      }
      MPSGraphTensor* argSortedTensor = [mpsGraph argSortWithTensor:castInputTensor
                                                               axis:(NSInteger)dim
                                                         descending:(BOOL)descending
                                                               name:@"argsort_out"];
      if ([argSortedTensor dataType] != getMPSDataType(indices)) {
        argSortedTensor = castMPSTensor(mpsGraph, argSortedTensor, indices.scalar_type());
      }
      newCachedGraph->valuesTensor = sortedTensor;
      newCachedGraph->indicesTensor = argSortedTensor;
    });
    Placeholder inputPlaceholder = Placeholder(cachedGraph->selfTensor, self);
    // Outputs as placeholders
    Placeholder valuesPlaceholder = Placeholder(cachedGraph->valuesTensor, values);
    Placeholder indicesPlaceholder = Placeholder(cachedGraph->indicesTensor, indices);
    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    auto results = dictionaryFromPlaceholders(valuesPlaceholder, indicesPlaceholder);

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}
} // namespace at::native
