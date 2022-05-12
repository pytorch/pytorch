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

    c10::MaybeOwned<Tensor> min_maybe_owned =
    at::borrow_from_optional_tensor(min_opt.get());
    c10::MaybeOwned<Tensor> max_maybe_owned =
    at::borrow_from_optional_tensor(max_opt.get());

    TORCH_CHECK(has_min || has_max, op_name + ": either min, max or both tensors must be defined")
    if (has_min)
        TORCH_CHECK(min_maybe_owned->is_same_size(input_t), op_name + ": min and input tensors must be of the same shape")
    if (has_max)
        TORCH_CHECK(max_maybe_owned->is_same_size(input_t), op_name + ": max and input tensors must be of the same shape")

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
                        newCachedGraph->minTensor = mpsGraphRankedPlaceHolder(mpsGraph, *min_maybe_owned);
                    if (has_max)
                        newCachedGraph->maxTensor = mpsGraphRankedPlaceHolder(mpsGraph, *max_maybe_owned);

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
            auto minPlaceholder = Placeholder(cachedGraph->minTensor, *min_maybe_owned);
            feeds[minPlaceholder.getMPSGraphTensor()] = minPlaceholder.getMPSGraphTensorData();
        }
        if (has_max) {
            auto maxPlaceholder = Placeholder(cachedGraph->maxTensor, *max_maybe_owned);
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

// Templates don't work with the build system, so there are two versions (Tensor/Scalar) for each clamp_x op
Tensor clamp_mps(const Tensor& input_t, const c10::optional<Tensor>& min, const c10::optional<Tensor>& max)
{
    Tensor result = at::empty({0}, input_t.options());
    mps::clamp_tensor_out_mps(input_t, min, max, result, __func__);
    return result;
}

Tensor clamp_mps(const Tensor& input_t, const OptionalScalarRef min, const OptionalScalarRef max)
{
    Tensor result = at::empty({0}, input_t.options());
    mps::clamp_scalar_out_mps(input_t, min, max, result, __func__);
    return result;
}

TORCH_IMPL_FUNC(clamp_min_Tensor_out_mps)
(const Tensor& input_t, const Tensor& min, const Tensor& output_t)
{
    mps::clamp_tensor_out_mps(input_t, min, c10::nullopt, output_t, __func__);
}

TORCH_IMPL_FUNC(clamp_min_out_mps)
(const Tensor& input_t, const Scalar& min, const Tensor& output_t)
{
    mps::clamp_scalar_out_mps(input_t, min, at::OptionalScalarRef(), output_t, __func__);
}

TORCH_IMPL_FUNC(clamp_max_Tensor_out_mps)
(const Tensor& input_t, const Tensor& max, const Tensor& output_t)
{
    mps::clamp_tensor_out_mps(input_t, c10::nullopt, max, output_t, __func__);
}

TORCH_IMPL_FUNC(clamp_max_out_mps)
(const Tensor& input_t, const Scalar& max, const Tensor& output_t)
{
    mps::clamp_scalar_out_mps(input_t, at::OptionalScalarRef(), max, output_t, __func__);
}
}
}
