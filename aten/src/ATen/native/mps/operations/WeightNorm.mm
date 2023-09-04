#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_weight_norm_interface_backward_native.h>
#include <ATen/ops/_weight_norm_interface_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#endif

namespace at::native {

using namespace at::native::mps;

// Derive from MPSCachedGraph
struct WeightNormCachedGraph : public MPSCachedGraph {
  WeightNormCachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* g_ = nil;
  MPSGraphTensor* v_ = nil;
  MPSGraphTensor* norms_ = nil;
  MPSGraphTensor* w_ = nil;
};

struct WeightNormBackwardCachedGraph : public MPSCachedGraph {
  WeightNormBackwardCachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* grad_w = nil;
  MPSGraphTensor* saved_v = nil;
  MPSGraphTensor* saved_g = nil;
  MPSGraphTensor* saved_norms = nil;
  MPSGraphTensor* grad_g = nil;
  MPSGraphTensor* grad_v = nil;
};

std::tuple<Tensor, Tensor> weight_norm_mps(const Tensor& v, const Tensor& g, int64_t dim) {
  MPSStream* mpsStream = getCurrentMPSStream();

  auto w = at::empty_like(v, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto norms = at::empty_strided(g.sizes(), g.strides(), g.options().dtype(g.scalar_type()));

  const int ndims = v.dim();

  if (dim == 0) {
    @autoreleasepool {
      string key = "weight_norm_mps_first_dim" + getTensorsStringKey({v, g});
      auto cachedGraph = LookUpOrCreateCachedGraph<WeightNormCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
        // Placeholders
        newCachedGraph->v_ = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(v.scalar_type()), getMPSShape(v));
        newCachedGraph->g_ = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(g.scalar_type()), getMPSShape(g));

        // Compute the L2 norm for each column of v
        MPSGraphTensor* squared = [mpsGraph squareWithTensor:newCachedGraph->v_ name:nil];
        MPSGraphTensor* sum_squared = [mpsGraph reductionSumWithTensor:squared axes:@[ @1 ] name:nil];
        newCachedGraph->norms_ = [mpsGraph squareRootWithTensor:sum_squared name:nil];

        // Divide each column of v by its L2 norm
        MPSGraphTensor* unit_v = [mpsGraph divisionWithPrimaryTensor:newCachedGraph->v_
                                                     secondaryTensor:newCachedGraph->norms_
                                                                name:nil];

        // Multiply each columb of vNormalized by the corresponding element of g
        newCachedGraph->w_ = [mpsGraph multiplicationWithPrimaryTensor:unit_v
                                                       secondaryTensor:newCachedGraph->g_
                                                                  name:nil];
      });

      Placeholder v_placeholder = Placeholder(cachedGraph->v_, v, nil, true);
      Placeholder g_placeholder = Placeholder(cachedGraph->g_, g, nil, true);
      Placeholder norms_placeholder = Placeholder(cachedGraph->norms_, norms);
      Placeholder w_placeholder = Placeholder(cachedGraph->w_, w);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
        v_placeholder.getMPSGraphTensor() : v_placeholder.getMPSGraphTensorData(),
        g_placeholder.getMPSGraphTensor() : g_placeholder.getMPSGraphTensorData()
      };

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
        norms_placeholder.getMPSGraphTensor() : norms_placeholder.getMPSGraphTensorData(),
        w_placeholder.getMPSGraphTensor() : w_placeholder.getMPSGraphTensorData()
      };

      runMPSGraph(mpsStream, cachedGraph->graph(), feeds, results);
    }
  } else if (dim == ndims - 1) {
    @autoreleasepool {
      string key = "weight_norm_mps_last_dim" + getTensorsStringKey({v, g});
      auto cachedGraph = LookUpOrCreateCachedGraph<WeightNormCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
        // Placeholders
        newCachedGraph->v_ = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(v.scalar_type()), getMPSShape(v));
        newCachedGraph->g_ = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(g.scalar_type()), getMPSShape(g));

        // Compute the L2 norm for each column of v
        MPSGraphTensor* squared = [mpsGraph squareWithTensor:newCachedGraph->v_ name:nil];
        MPSGraphTensor* sum_squared = [mpsGraph reductionSumWithTensor:squared axes:@[ @0 ] name:nil];
        newCachedGraph->norms_ = [mpsGraph squareRootWithTensor:sum_squared name:nil];

        // Divide each column of v by its L2 norm
        MPSGraphTensor* unit_v = [mpsGraph divisionWithPrimaryTensor:newCachedGraph->v_
                                                     secondaryTensor:newCachedGraph->norms_
                                                                name:nil];

        // Multiply each columb of vNormalized by the corresponding element of g
        newCachedGraph->w_ = [mpsGraph multiplicationWithPrimaryTensor:unit_v
                                                       secondaryTensor:newCachedGraph->g_
                                                                  name:nil];
      });

      Placeholder v_placeholder = Placeholder(cachedGraph->v_, v, nil, true);
      Placeholder g_placeholder = Placeholder(cachedGraph->g_, g, nil, true);
      Placeholder norms_placeholder = Placeholder(cachedGraph->norms_, norms);
      Placeholder w_placeholder = Placeholder(cachedGraph->w_, w);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
        v_placeholder.getMPSGraphTensor() : v_placeholder.getMPSGraphTensorData(),
        g_placeholder.getMPSGraphTensor() : g_placeholder.getMPSGraphTensorData()
      };

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
        norms_placeholder.getMPSGraphTensor() : norms_placeholder.getMPSGraphTensorData(),
        w_placeholder.getMPSGraphTensor() : w_placeholder.getMPSGraphTensorData()
      };

      runMPSGraph(mpsStream, cachedGraph->graph(), feeds, results);
    }
  }

  return std::tuple<Tensor, Tensor>{w, norms};
}

std::tuple<Tensor, Tensor> weight_norm_backward_mps(const Tensor& grad_w,
                                                    const Tensor& saved_v,
                                                    const Tensor& saved_g,
                                                    const Tensor& saved_norms,
                                                    int64_t dim) {
  // These checks should always succeed, because weight_norm_fused_backward should only
  // ever be recorded in the autograd graph via weight_norm, which passes contiguous v and g.
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  TORCH_CHECK(saved_norms.is_contiguous(), "saved_norms must be contiguous");
  TORCH_CHECK(dim == 0 || dim == saved_v.dim() - 1, "fused kernels can only be applied for first or last dim")

  auto grad_v = at::empty_like(saved_v, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_g = at::empty_like(saved_g, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  MPSStream* mpsStream = getCurrentMPSStream();

  const int ndims = saved_v.dim();

  if (dim == 0) {
    @autoreleasepool {
      string key = "weight_norm_backward_mps_first_dim" + getTensorsStringKey({grad_w, saved_v, saved_g, saved_norms});
      auto cachedGraph =
          LookUpOrCreateCachedGraph<WeightNormBackwardCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
            // Placeholders
            newCachedGraph->grad_w =
                mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(grad_w.scalar_type()), getMPSShape(grad_w));
            newCachedGraph->saved_v =
                mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(saved_v.scalar_type()), getMPSShape(saved_v));
            newCachedGraph->saved_g =
                mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(saved_g.scalar_type()), getMPSShape(saved_g));
            newCachedGraph->saved_norms = mpsGraphRankedPlaceHolder(
                mpsGraph, getMPSScalarType(saved_norms.scalar_type()), getMPSShape(saved_norms));

            // Compute Graph
            MPSGraphTensor* grad_w_v = [mpsGraph multiplicationWithPrimaryTensor:newCachedGraph->grad_w
                                                                 secondaryTensor:newCachedGraph->saved_v
                                                                            name:nil];
            MPSGraphTensor* result = [mpsGraph reductionSumWithTensor:grad_w_v axes:@[ @1 ] name:nil];

            newCachedGraph->grad_g = [mpsGraph divisionWithPrimaryTensor:result
                                                         secondaryTensor:newCachedGraph->saved_norms
                                                                    name:nil];

            MPSGraphTensor* grad_w_divided_by_norm = [mpsGraph divisionWithPrimaryTensor:newCachedGraph->grad_w
                                                                         secondaryTensor:newCachedGraph->saved_norms
                                                                                    name:nil];
            MPSGraphTensor* three = [mpsGraph constantWithScalar:3.0 dataType:newCachedGraph->saved_norms.dataType];
            MPSGraphTensor* norm_cubed = [mpsGraph powerWithPrimaryTensor:newCachedGraph->saved_norms
                                                          secondaryTensor:three
                                                                     name:nil];
            MPSGraphTensor* v_result = [mpsGraph multiplicationWithPrimaryTensor:newCachedGraph->saved_v
                                                                 secondaryTensor:result
                                                                            name:nil];
            MPSGraphTensor* v_result_divided_by_norm_cubed = [mpsGraph divisionWithPrimaryTensor:v_result
                                                                                 secondaryTensor:norm_cubed
                                                                                            name:nil];
            MPSGraphTensor* diff = [mpsGraph subtractionWithPrimaryTensor:grad_w_divided_by_norm
                                                          secondaryTensor:v_result_divided_by_norm_cubed
                                                                     name:nil];
            newCachedGraph->grad_v = [mpsGraph multiplicationWithPrimaryTensor:diff
                                                               secondaryTensor:newCachedGraph->saved_g
                                                                          name:nil];
          });

      Placeholder grad_w_placeholder = Placeholder(cachedGraph->grad_w, grad_w, nil, true);
      Placeholder v_placeholder = Placeholder(cachedGraph->saved_v, saved_v, nil, true);
      Placeholder g_placeholder = Placeholder(cachedGraph->saved_g, saved_g, nil, true);
      Placeholder norms_placeholder = Placeholder(cachedGraph->saved_norms, saved_norms, nil, true);

      Placeholder grad_g_placeholder = Placeholder(cachedGraph->grad_g, grad_g);
      Placeholder grad_v_placeholder = Placeholder(cachedGraph->grad_v, grad_v);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
        grad_w_placeholder.getMPSGraphTensor() : grad_w_placeholder.getMPSGraphTensorData(),
        norms_placeholder.getMPSGraphTensor() : norms_placeholder.getMPSGraphTensorData(),
        v_placeholder.getMPSGraphTensor() : v_placeholder.getMPSGraphTensorData(),
        g_placeholder.getMPSGraphTensor() : g_placeholder.getMPSGraphTensorData()
      };

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
        grad_g_placeholder.getMPSGraphTensor() : grad_g_placeholder.getMPSGraphTensorData(),
        grad_v_placeholder.getMPSGraphTensor() : grad_v_placeholder.getMPSGraphTensorData()
      };

      runMPSGraph(mpsStream, cachedGraph->graph(), feeds, results);
    }
  } else if (dim == ndims - 1) {
    @autoreleasepool {
      string key = "weight_norm_backward_mps_last_dim" + getTensorsStringKey({grad_w, saved_v, saved_g, saved_norms});
      auto cachedGraph =
          LookUpOrCreateCachedGraph<WeightNormBackwardCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
            // Placeholders
            newCachedGraph->grad_w =
                mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(grad_w.scalar_type()), getMPSShape(grad_w));
            newCachedGraph->saved_v =
                mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(saved_v.scalar_type()), getMPSShape(saved_v));
            newCachedGraph->saved_g =
                mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(saved_g.scalar_type()), getMPSShape(saved_g));
            newCachedGraph->saved_norms = mpsGraphRankedPlaceHolder(
                mpsGraph, getMPSScalarType(saved_norms.scalar_type()), getMPSShape(saved_norms));

            // Compute Graph
            MPSGraphTensor* grad_w_v = [mpsGraph multiplicationWithPrimaryTensor:newCachedGraph->grad_w
                                                                 secondaryTensor:newCachedGraph->saved_v
                                                                            name:nil];
            MPSGraphTensor* result = [mpsGraph reductionSumWithTensor:grad_w_v axes:@[ @0 ] name:nil];

            newCachedGraph->grad_g = [mpsGraph divisionWithPrimaryTensor:result
                                                         secondaryTensor:newCachedGraph->saved_norms
                                                                    name:nil];

            MPSGraphTensor* grad_w_divided_by_norm = [mpsGraph divisionWithPrimaryTensor:newCachedGraph->grad_w
                                                                         secondaryTensor:newCachedGraph->saved_norms
                                                                                    name:nil];
            MPSGraphTensor* three = [mpsGraph constantWithScalar:3.0 dataType:newCachedGraph->saved_norms.dataType];
            MPSGraphTensor* norm_cubed = [mpsGraph powerWithPrimaryTensor:newCachedGraph->saved_norms
                                                          secondaryTensor:three
                                                                     name:nil];
            MPSGraphTensor* v_result = [mpsGraph multiplicationWithPrimaryTensor:newCachedGraph->saved_v
                                                                 secondaryTensor:result
                                                                            name:nil];
            MPSGraphTensor* v_result_divided_by_norm_cubed = [mpsGraph divisionWithPrimaryTensor:v_result
                                                                                 secondaryTensor:norm_cubed
                                                                                            name:nil];
            MPSGraphTensor* diff = [mpsGraph subtractionWithPrimaryTensor:grad_w_divided_by_norm
                                                          secondaryTensor:v_result_divided_by_norm_cubed
                                                                     name:nil];
            newCachedGraph->grad_v = [mpsGraph multiplicationWithPrimaryTensor:diff
                                                               secondaryTensor:newCachedGraph->saved_g
                                                                          name:nil];
          });

      Placeholder grad_w_placeholder = Placeholder(cachedGraph->grad_w, grad_w, nil, true);
      Placeholder v_placeholder = Placeholder(cachedGraph->saved_v, saved_v, nil, true);
      Placeholder g_placeholder = Placeholder(cachedGraph->saved_g, saved_g, nil, true);
      Placeholder norms_placeholder = Placeholder(cachedGraph->saved_norms, saved_norms, nil, true);

      Placeholder grad_g_placeholder = Placeholder(cachedGraph->grad_g, grad_g);
      Placeholder grad_v_placeholder = Placeholder(cachedGraph->grad_v, grad_v);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
        grad_w_placeholder.getMPSGraphTensor() : grad_w_placeholder.getMPSGraphTensorData(),
        norms_placeholder.getMPSGraphTensor() : norms_placeholder.getMPSGraphTensorData(),
        v_placeholder.getMPSGraphTensor() : v_placeholder.getMPSGraphTensorData(),
        g_placeholder.getMPSGraphTensor() : g_placeholder.getMPSGraphTensorData()
      };

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
        grad_g_placeholder.getMPSGraphTensor() : grad_g_placeholder.getMPSGraphTensorData(),
        grad_v_placeholder.getMPSGraphTensor() : grad_v_placeholder.getMPSGraphTensorData()
      };

      runMPSGraph(mpsStream, cachedGraph->graph(), feeds, results);
    }
  }

  return std::tuple<Tensor, Tensor>{grad_v, grad_g};
}

} // namespace at::native
