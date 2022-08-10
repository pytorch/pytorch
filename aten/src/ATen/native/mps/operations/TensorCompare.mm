//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/TensorUtils.h>

namespace at {
namespace native {
namespace mps {

struct CachedGraph : public MPSCachedGraph
{
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor = nil, *outputTensor = nil;
    MPSGraphTensor *minTensor = nil, *maxTensor = nil;
};

void clamp_mps_graph(CachedGraph* cachedGraph, const Tensor& input_tensor)
{
    MPSGraph *mpsGraph = cachedGraph->graph();

    cachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_tensor);

    if (cachedGraph->minTensor && cachedGraph->maxTensor) {
        cachedGraph->outputTensor = [mpsGraph clampWithTensor:cachedGraph->inputTensor
                                               minValueTensor:cachedGraph->minTensor
                                               maxValueTensor:cachedGraph->maxTensor
                                                         name:nil];
    } else if (cachedGraph->maxTensor) {
        cachedGraph->outputTensor = [mpsGraph minimumWithPrimaryTensor:cachedGraph->inputTensor
                                                       secondaryTensor:cachedGraph->maxTensor
                                                                  name:nil];
    } else if (cachedGraph->minTensor) {
        cachedGraph->outputTensor = [mpsGraph maximumWithPrimaryTensor:cachedGraph->inputTensor
                                                       secondaryTensor:cachedGraph->minTensor
                                                                  name:nil];
    }
}

void clamp_tensor_out_mps(const Tensor& input_t,
                          const OptionalTensorRef min_opt,
                          const OptionalTensorRef max_opt,
                          const Tensor& output_t,
                          string op_name)
{
    const bool has_min = (min_opt.has_value() && min_opt->defined());
    const bool has_max = (max_opt.has_value() && max_opt->defined());

    TORCH_CHECK(has_min || has_max, op_name + ": either min, max or both tensors must be defined")
    if (has_min)
        TORCH_CHECK(min_opt->is_same_size(input_t), op_name + ": min and input tensors must be of the same shape")
    if (has_max)
        TORCH_CHECK(max_opt->is_same_size(input_t), op_name + ": max and input tensors must be of the same shape")

    if (output_t.numel() == 0)
        return;

    @autoreleasepool {
        // the optional min/max refs could affect how we build the cached graph
        string key = op_name + (has_min ? "_min" : "") + (has_max ? "_max" : "")
                             + "_tensor" + getTensorsStringKey({input_t});
        MPSGraphCache* cache_ = MPSGraphCache::getInstance();
        CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

        if (!cachedGraph) {
            MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
                CachedGraph *newCachedGraph = nil;

                @autoreleasepool {
                    MPSGraph* mpsGraph = make_mps_graph();
                    newCachedGraph = new CachedGraph(mpsGraph);

                    if (has_min)
                        newCachedGraph->minTensor = mpsGraphRankedPlaceHolder(mpsGraph, *min_opt);
                    if (has_max)
                        newCachedGraph->maxTensor = mpsGraphRankedPlaceHolder(mpsGraph, *max_opt);

                    clamp_mps_graph(newCachedGraph, input_t);
                }
                return newCachedGraph;
            });
            cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
        }

        auto inputPlaceholder  = Placeholder(cachedGraph->inputTensor, input_t);
        auto outputPlaceholder = Placeholder(cachedGraph->outputTensor, output_t);

        NSMutableDictionary *feeds = [[NSMutableDictionary new] autorelease];
        feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
        if (has_min) {
            auto minPlaceholder = Placeholder(cachedGraph->minTensor, *min_opt);
            feeds[minPlaceholder.getMPSGraphTensor()] = minPlaceholder.getMPSGraphTensorData();
        }
        if (has_max) {
            auto maxPlaceholder = Placeholder(cachedGraph->maxTensor, *max_opt);
            feeds[maxPlaceholder.getMPSGraphTensor()] = maxPlaceholder.getMPSGraphTensorData();
        }

        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
            outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
        };

        runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, results);
    }
}

void clamp_scalar_out_mps(const Tensor& input_t,
                             const OptionalScalarRef min_opt,
                             const OptionalScalarRef max_opt,
                             const Tensor& output_t,
                             string op_name)
{
    using scalar_t = double;

    const bool has_min = (min_opt.has_value());
    const bool has_max = (max_opt.has_value());
    TORCH_CHECK(has_min || has_max, op_name + ": either min, max or both scalars must be defined")

    scalar_t min_scalar =  std::numeric_limits<scalar_t>::infinity();
    scalar_t max_scalar = -std::numeric_limits<scalar_t>::infinity();

    if (has_min)
        min_scalar = min_opt.get().to<scalar_t>();
    if (has_max)
        max_scalar = max_opt.get().to<scalar_t>();

    if (output_t.numel() == 0)
        return ;

    @autoreleasepool {
        // the optional min/max refs could affect how we build the cached graph
        string key = op_name + (has_min ? ("_min:" + to_string(min_scalar)) : "") + (has_max ? ("_max:" + to_string(max_scalar)) : "")
                             + "_scalar:" + getTensorsStringKey({input_t});
        MPSGraphCache* cache_ = MPSGraphCache::getInstance();
        CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

        if (!cachedGraph) {
            MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
                CachedGraph *newCachedGraph = nil;

                @autoreleasepool {
                    MPSGraph* mpsGraph = make_mps_graph();
                    newCachedGraph = new CachedGraph(mpsGraph);

                    if (has_min)
                        newCachedGraph->minTensor = [mpsGraph constantWithScalar:min_scalar
                                                                           shape:(mps::getMPSShape(input_t))
                                                                        dataType:(mps::getMPSScalarType(input_t.scalar_type())) ];
                    if (has_max)
                        newCachedGraph->maxTensor = [mpsGraph constantWithScalar:max_scalar
                                                                           shape:(mps::getMPSShape(input_t))
                                                                        dataType:(mps::getMPSScalarType(input_t.scalar_type())) ];

                    clamp_mps_graph(newCachedGraph, input_t);
                }
                return newCachedGraph;
            });
            cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
        }

        auto inputPlaceholder  = Placeholder(cachedGraph->inputTensor , input_t);
        auto outputPlaceholder = Placeholder(cachedGraph->outputTensor, output_t);

        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
          inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
        };
        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
            outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
        };

        runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, results);
    }
}

} // namespace mps

// APIs exposed to at::native scope
TORCH_IMPL_FUNC(clamp_Tensor_out_mps)
(const Tensor& input_t, const OptionalTensorRef min, const OptionalTensorRef max, const Tensor& output_t)
{
    mps::clamp_tensor_out_mps(input_t, min, max, output_t, __func__);
}

TORCH_IMPL_FUNC(clamp_out_mps)
(const Tensor& input_t, const OptionalScalarRef min, const OptionalScalarRef max, const Tensor& output_t)
{
    mps::clamp_scalar_out_mps(input_t, min, max, const_cast<Tensor&>(output_t), "clamp_out_mps");
}

TORCH_IMPL_FUNC(clamp_min_Tensor_out_mps)
(const Tensor& input_t, const Tensor& min, const Tensor& output_t)
{
    mps::clamp_tensor_out_mps(input_t, min, at::OptionalTensorRef(), output_t, __func__);
}

TORCH_IMPL_FUNC(clamp_min_out_mps)
(const Tensor& input_t, const Scalar& min, const Tensor& output_t)
{
    mps::clamp_scalar_out_mps(input_t, min, at::OptionalScalarRef(), output_t, __func__);
}

TORCH_IMPL_FUNC(clamp_max_Tensor_out_mps)
(const Tensor& input_t, const Tensor& max, const Tensor& output_t)
{
    mps::clamp_tensor_out_mps(input_t, at::OptionalTensorRef(), max, output_t, __func__);
}

TORCH_IMPL_FUNC(clamp_max_out_mps)
(const Tensor& input_t, const Scalar& max, const Tensor& output_t)
{
    mps::clamp_scalar_out_mps(input_t, at::OptionalScalarRef(), max, output_t, __func__);
}

Tensor& where_self_out_mps(const Tensor& condition,
                           const Tensor& self,
                           const Tensor& other,
                           Tensor& out) {
  TORCH_CHECK(self.dtype() == other.dtype(), "expected scalar type ", self.dtype(), " but found ", other.dtype());

  if (condition.scalar_type() == ScalarType::Byte) {
  TORCH_WARN_ONCE("where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead.");
  } else {
  TORCH_CHECK(condition.scalar_type() == ScalarType::Bool, "where expected condition to be a boolean tensor, but got a tensor with dtype ", condition.scalar_type());
  }
  Tensor cond_bool = condition.scalar_type() == ScalarType::Byte ? condition.to(ScalarType::Bool) : condition;

  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();

  // Empty output
  if(out.numel() == 0)
    return out;

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* conditionTensor_ = nil;
    MPSGraphTensor* selfTensor_ = nil;
    MPSGraphTensor* otherTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {

    string key = "where_self_out_mps:" + getTensorsStringKey({cond_bool, self, other});

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
        MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

            CachedGraph *newCachedGraph = nil;

            @autoreleasepool {
                MPSGraph* mpsGraph = make_mps_graph();
                newCachedGraph = new CachedGraph(mpsGraph);

                MPSGraphTensor* conditionTensor = mpsGraphRankedPlaceHolder(mpsGraph, cond_bool);
                MPSGraphTensor* selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
                MPSGraphTensor* otherTensor = mpsGraphRankedPlaceHolder(mpsGraph, other);

                MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:conditionTensor
                                                               truePredicateTensor:selfTensor
                                                              falsePredicateTensor:otherTensor
                                                                              name:nil];

                newCachedGraph->conditionTensor_ = conditionTensor;
                newCachedGraph->selfTensor_ = selfTensor;
                newCachedGraph->otherTensor_ = otherTensor;
                newCachedGraph->outputTensor_ = outputTensor;
            }
            return newCachedGraph;
        });
        cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder conditionPlaceholder = Placeholder(cachedGraph->conditionTensor_, cond_bool);
    Placeholder selfPlaceholder = Placeholder(cachedGraph->selfTensor_, self);
    Placeholder otherPlaceholder = Placeholder(cachedGraph->otherTensor_, other);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      conditionPlaceholder.getMPSGraphTensor() : conditionPlaceholder.getMPSGraphTensorData(),
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      otherPlaceholder.getMPSGraphTensor() : otherPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);

  }

  return out;
}

Tensor where_mps(const Tensor& condition,
                 const Tensor& self,
                 const Tensor& other) {

  bool cond_zero_shape = (condition.dim() == 0);
  bool self_zero_shape = (self.dim() == 0);
  bool other_zero_shape = (other.dim() == 0);

  auto max_dim = std::max(condition.dim(), std::max(self.dim(), other.dim()));

  auto sum_dims = condition.dim() + self.dim() + other.dim();

  TORCH_CHECK(max_dim == 0 || !(sum_dims % max_dim), "All inputs of where should have same/compatible number of dims")

  int64_t out_arr[max_dim];

  // Broadcasted output shape
  for(int i = 0; i < max_dim; i++) {

    int64_t cond_num = cond_zero_shape ? 0 : condition.size(i);
    int64_t self_num = self_zero_shape ? 0 : self.size(i);
    int64_t other_num = other_zero_shape ? 0 : other.size(i);

    out_arr[i] = std::max(cond_num, std::max(self_num, other_num));
  }

  Tensor ret = empty_mps(IntArrayRef(out_arr, max_dim),
                         self.scalar_type(),
                         c10::nullopt,
                         kMPS,
                         c10::nullopt,
                         self.suggest_memory_format());
  return where_self_out_mps(condition, self, other, ret);

}

} // namespace native
} // namespace at
