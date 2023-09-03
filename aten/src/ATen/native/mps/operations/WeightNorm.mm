#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/_weight_norm_interface_native.h>
#include <ATen/ops/_weight_norm_interface_backward_native.h>
#endif

namespace at::native {

using namespace at::native::mps;

// Derive from MPSCachedGraph
struct CachedGraph : public MPSCachedGraph {
  CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* g_ = nil;
  MPSGraphTensor* v_ = nil;
  MPSGraphTensor* norms_ = nil;
  MPSGraphTensor* w_ = nil;
};

std::tuple<Tensor,Tensor> weight_norm_mps
  (const Tensor& v,
   const Tensor& g,
   int64_t dim)
{
  MPSStream* mpsStream = getCurrentMPSStream();

  auto w = at::empty_like(v, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  auto norms = at::empty_strided(g.sizes(), g.strides(), g.options().dtype(g.scalar_type()));

  const int ndims = v.dim();

  if(dim == 0)
  {
    @autoreleasepool {
      string key = "weight_norm_mps_first_dim" + getTensorsStringKey({v, g});
      auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {

        newCachedGraph->v_ = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(v.scalar_type()), getMPSShape(v));
        newCachedGraph->g_ = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(g.scalar_type()), getMPSShape(g));

        // Compute the L2 norm for each row of v
        MPSGraphTensor *squared = [mpsGraph squareWithTensor:newCachedGraph->v_ name:nil];
        MPSGraphTensor *sum_squared = [mpsGraph reductionSumWithTensor:squared axes:@[@1] name:nil];
        MPSGraphTensor *norms_ = [mpsGraph squareRootWithTensor:sum_squared name:nil];
        newCachedGraph->norms_ = norms_;
        // Divide each row of v by its L2 norm
        MPSGraphTensor *unit_v = [mpsGraph divisionWithPrimaryTensor:newCachedGraph->v_ secondaryTensor:norms_ name:nil];

        // Multiply each row of vNormalized by the corresponding element of g
        newCachedGraph->w_ = [mpsGraph multiplicationWithPrimaryTensor:unit_v secondaryTensor:newCachedGraph->g_ name:nil];
      });

      Placeholder vPlaceholder = Placeholder(cachedGraph->v_, v, nil, true);
      Placeholder gPlaceholder = Placeholder(cachedGraph->g_, g, nil, true);
      Placeholder normsPlaceholder = Placeholder(cachedGraph->norms_, norms);
      Placeholder wPlaceholder = Placeholder(cachedGraph->w_, w);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
          @{vPlaceholder.getMPSGraphTensor() : vPlaceholder.getMPSGraphTensorData(),
            gPlaceholder.getMPSGraphTensor() : gPlaceholder.getMPSGraphTensorData()};

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
          @{normsPlaceholder.getMPSGraphTensor() : normsPlaceholder.getMPSGraphTensorData(),
            wPlaceholder.getMPSGraphTensor() : wPlaceholder.getMPSGraphTensorData()};

      runMPSGraph(mpsStream, cachedGraph->graph(), feeds, results);
    }
  }
  else if(dim == ndims - 1)
  {
    @autoreleasepool {
      string key = "weight_norm_mps_second_dim" + getTensorsStringKey({v, g});
      auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {

        newCachedGraph->v_ = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(v.scalar_type()), getMPSShape(v));
        newCachedGraph->g_ = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(g.scalar_type()), getMPSShape(g));

        // Compute the L2 norm for each column of v
        MPSGraphTensor *squared = [mpsGraph squareWithTensor:newCachedGraph->v_ name:nil];
        MPSGraphTensor *sum_squared = [mpsGraph reductionSumWithTensor:squared axes:@[@0] name:nil];
        MPSGraphTensor *norms_ = [mpsGraph squareRootWithTensor:sum_squared name:nil];
        newCachedGraph->norms_ = norms_;
        // Divide each column of v by its L2 norm
        MPSGraphTensor *unit_v = [mpsGraph divisionWithPrimaryTensor:newCachedGraph->v_ secondaryTensor:norms_ name:nil];

        // Multiply each columb of vNormalized by the corresponding element of g
        newCachedGraph->w_ = [mpsGraph multiplicationWithPrimaryTensor:unit_v secondaryTensor:newCachedGraph->g_ name:nil];
      });

      Placeholder vPlaceholder = Placeholder(cachedGraph->v_, v, nil, true);
      Placeholder gPlaceholder = Placeholder(cachedGraph->g_, g, nil, true);
      Placeholder normsPlaceholder = Placeholder(cachedGraph->norms_, norms);
      Placeholder wPlaceholder = Placeholder(cachedGraph->w_, w);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
          @{vPlaceholder.getMPSGraphTensor() : vPlaceholder.getMPSGraphTensorData(),
            gPlaceholder.getMPSGraphTensor() : gPlaceholder.getMPSGraphTensorData()};

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
          @{normsPlaceholder.getMPSGraphTensor() : normsPlaceholder.getMPSGraphTensorData(),
            wPlaceholder.getMPSGraphTensor() : wPlaceholder.getMPSGraphTensorData()};

      runMPSGraph(mpsStream, cachedGraph->graph(), feeds, results);
    }
  }

  return std::tuple<Tensor, Tensor>{w, norms};
}

}
