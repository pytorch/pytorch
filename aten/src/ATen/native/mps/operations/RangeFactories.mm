//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/AccumulateType.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <cmath>
#include <limits>

namespace at {
namespace native {


Tensor& arange_mps_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
  AT_DISPATCH_MPS_TYPES(result.scalar_type(), "arange_mps", [&]() {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    double size_d;
    if (std::is_same<scalar_t, int64_t>::value) {
      size_d = std::ceil(static_cast<double>(end.to<accscalar_t>() - start.to<accscalar_t>())
                          / step.to<accscalar_t>());
    } else {
      size_d = std::ceil(static_cast<double>(end.to<double>() - start.to<double>())
                          / step.to<double>());
    }

    TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
    TORCH_CHECK(std::isfinite(static_cast<double>(xstart)) &&
              std::isfinite(static_cast<double>(xend)),
              "unsupported range: ", xstart, " -> ", xend);
    TORCH_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
              "upper bound and larger bound inconsistent with step sign");

    TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
              "invalid size, possible overflow?");
    int64_t size = static_cast<int64_t>(size_d);
    int64_t numel = result.numel();

    if (numel != size) {
      if(numel > 0){
        TORCH_WARN("The number of elements in the out tensor of shape ", result.sizes(),
                    " is ", numel, " which does not match the computed number of elements ", size,
                    ". Note that this may occur as a result of rounding error. "
                    "The out tensor will be resized to a tensor of shape (", size, ",).");
      }
      result.resize_({size});
    }
    bool is_contiguous = result.is_contiguous();
    Tensor r = !is_contiguous ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;
    using namespace mps;
    struct CachedGraph : public MPSCachedGraph {
      CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
      MPSGraphTensor *startTensor = nil;
      MPSGraphTensor *multiplyTensor = nil;
      MPSGraphTensor *outputTensor = nil;
    };
    auto cache_ = MPSGraphCache::getInstance();
    auto stream = getCurrentMPSStream();
    auto mpsDataType = getMPSDataType(result.scalar_type());
    @autoreleasepool {
      string key = "arange_mps_out:" + getTensorsStringKey({result}) + ":" + to_string(size);
      auto cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
      if (!cachedGraph) {
        auto *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph *() {
          CachedGraph *newCachedGraph = nil;
          @autoreleasepool {
            auto mpsGraph = make_mps_graph();
            newCachedGraph = new CachedGraph(mpsGraph);
            auto shapeTensor = [mpsGraph constantWithData:[NSData dataWithBytes:&size length:sizeof(size)]
                                                    shape: @[@1]
                                                 dataType:MPSDataTypeInt64];
            auto coordsTensor = [mpsGraph coordinateAlongAxis:0
                                              withShapeTensor:shapeTensor
                                                         name:nil];
            coordsTensor = [mpsGraph castTensor:coordsTensor toType:mpsDataType name:@"coords"];

            newCachedGraph->startTensor = mpsGraphRankedPlaceHolder(mpsGraph, mpsDataType, @[@1]);
            newCachedGraph->multiplyTensor = mpsGraphRankedPlaceHolder(mpsGraph, mpsDataType, @[@1]);
            auto scaledCoords = [mpsGraph multiplicationWithPrimaryTensor:coordsTensor
                                                          secondaryTensor:newCachedGraph->multiplyTensor
                                                                     name:nil];
            newCachedGraph->outputTensor = [mpsGraph additionWithPrimaryTensor:scaledCoords
                                                               secondaryTensor:newCachedGraph->startTensor
                                                                          name:nil];
          }

          return newCachedGraph;
        });
        cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
      }
      Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, r);
      NSMutableDictionary *feeds   = [[NSMutableDictionary new] autorelease];
      feeds[cachedGraph->startTensor] = getMPSGraphTensorFromScalar(stream, start, mpsDataType);
      feeds[cachedGraph->multiplyTensor] = getMPSGraphTensorFromScalar(stream, Scalar(step), mpsDataType);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
        outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
      };
      runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    }

    if(!is_contiguous) {
      result.copy_(r);
    }
  });

  return result;
}

Tensor& linspace_out_mps(const Scalar& start, const Scalar& end, int64_t steps, Tensor& result) {
  using namespace mps;
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *startTensor_ = nil;
    MPSGraphTensor *endTensor_ = nil;
    MPSGraphTensor *multiplyTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
  if (result.numel() != steps) {
    result.resize_({steps});
  }

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    result.fill_(start);
  } else {
    Tensor r = result.is_contiguous() ? result : result.contiguous();

    // Do the MPSGraph computation
    MPSGraphCache* cache_ = MPSGraphCache::getInstance();
    MPSStream* stream = getCurrentMPSStream();

    bool start_less_end = (start.to<double>() <= end.to<double>());

    @autoreleasepool {
      string key = "linspace_out_mps:" + getTensorsStringKey({result}) + ":" + to_string(steps) + to_string(start_less_end);
      CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

      if(!cachedGraph) {
        MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

          CachedGraph *newCachedGraph = nil;

          @autoreleasepool {
            MPSGraph* mpsGraph = make_mps_graph();
            newCachedGraph = new CachedGraph(mpsGraph);

            int shapeVal[1] = {(int32_t)steps};
            MPSGraphTensor *shapeTensor = [mpsGraph constantWithData:[NSData dataWithBytes:shapeVal length:sizeof(int32_t)]
                                                               shape: @[@1]
                                                              dataType:MPSDataTypeInt32];
            MPSGraphTensor* coordsTensor = [mpsGraph coordinateAlongAxis:0
                                                         withShapeTensor:shapeTensor
                                                                    name:nil];
            coordsTensor = [mpsGraph castTensor:coordsTensor toType:MPSDataTypeFloat32 name:@"coords"];

            MPSGraphTensor* startTensor = mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeFloat32, @[@1]);
            MPSGraphTensor* endTensor = mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeFloat32, @[@1]);
            MPSGraphTensor* multiplyTensor = mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeFloat32, @[@1]);
            MPSGraphTensor* scaledCoords = [mpsGraph multiplicationWithPrimaryTensor:coordsTensor
                                                                     secondaryTensor:multiplyTensor
                                                                                name:nil];
            MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:scaledCoords
                                                               secondaryTensor:startTensor
                                                                          name:nil];
            if(start_less_end) {
              outputTensor = [mpsGraph clampWithTensor:outputTensor
                                        minValueTensor:startTensor
                                        maxValueTensor:endTensor
                                                  name:nil];
            } else {
              outputTensor = [mpsGraph clampWithTensor:outputTensor
                                        minValueTensor:endTensor
                                        maxValueTensor:startTensor
                                                  name:nil];
            }

            if(getMPSDataType(result.scalar_type()) != MPSDataTypeFloat32) {
              outputTensor = [mpsGraph castTensor:outputTensor toType:getMPSDataType(result.scalar_type()) name:@"output"];
            }

            newCachedGraph->startTensor_ = startTensor;
            newCachedGraph->endTensor_ = endTensor;
            newCachedGraph->multiplyTensor_ = multiplyTensor;
            newCachedGraph->outputTensor_ = outputTensor;
          }
          return newCachedGraph;
        });
        cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
      }

      NSMutableDictionary *feeds   = [[NSMutableDictionary new] autorelease];
      auto multiplyScalar = (end.to<double>() - start.to<double>()) / ((double)steps - 1.0f);
      Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, r);

      // Create dictionary of inputs and outputs
      feeds[cachedGraph->startTensor_] = getMPSGraphTensorFromScalar(stream, start, MPSDataTypeFloat32);
      feeds[cachedGraph->endTensor_] = getMPSGraphTensorFromScalar(stream, end, MPSDataTypeFloat32);
      feeds[cachedGraph->multiplyTensor_] = getMPSGraphTensorFromScalar(stream, Scalar(multiplyScalar), MPSDataTypeFloat32);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
        outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
      };
      runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    }

    if (!result.is_contiguous()) {
      result.copy_(r);
    }
  }
  return result;
}
}} // namespace at::native
