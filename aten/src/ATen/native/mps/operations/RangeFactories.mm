//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/linspace_native.h>
#include <ATen/ops/range_native.h>
#include <cmath>
#include <limits>

namespace at::native {

namespace {
struct RangeCachedGraph : public mps::MPSCachedGraph {
  API_AVAILABLE(macosx(12.3))
  RangeCachedGraph(MPSGraph* mpsGraph,
                   MPSDataType dataType,
                   int32_t shapeVal,
                   bool needsClamp = false,
                   bool startLessEnd = false)
      : MPSCachedGraph(mpsGraph) {
    @autoreleasepool {
      auto shapeTensor = [mpsGraph constantWithData:[NSData dataWithBytes:&shapeVal length:sizeof(int32_t)]
                                              shape:@[ @1 ]
                                           dataType:MPSDataTypeInt32];
      auto coordsTensor = [mpsGraph coordinateAlongAxis:0 withShapeTensor:shapeTensor name:nil];
      coordsTensor = [mpsGraph castTensor:coordsTensor toType:dataType name:@"coords"];

      startTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, dataType, @[ @1 ]);
      multiplyTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, dataType, @[ @1 ]);
      auto scaledCoords = [mpsGraph multiplicationWithPrimaryTensor:coordsTensor
                                                    secondaryTensor:multiplyTensor
                                                               name:nil];
      outputTensor = [mpsGraph additionWithPrimaryTensor:scaledCoords secondaryTensor:startTensor name:nil];
      if (needsClamp) {
        endTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, dataType, @[ @1 ]);
        outputTensor = [mpsGraph clampWithTensor:outputTensor
                                  minValueTensor:startLessEnd ? startTensor : endTensor
                                  maxValueTensor:startLessEnd ? endTensor : startTensor
                                            name:nil];
      }
    }
  }
  MPSGraphTensor* startTensor = nil;
  MPSGraphTensor* endTensor = nil;
  MPSGraphTensor* multiplyTensor = nil;
  MPSGraphTensor* outputTensor = nil;
};

} // anonymous namespace

Tensor& arange_mps_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
  AT_DISPATCH_MPS_TYPES(result.scalar_type(), "arange_mps", [&]() {
    using accscalar_t = at::acc_type_device<scalar_t, kMPS>;
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    double size_d;
    if constexpr (std::is_same_v<scalar_t, int64_t>) {
      size_d = std::ceil(static_cast<double>(end.to<accscalar_t>() - start.to<accscalar_t>()) / step.to<accscalar_t>());
    } else {
      size_d = std::ceil(static_cast<double>(end.to<double>() - start.to<double>()) / step.to<double>());
    }

    TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
    TORCH_CHECK(std::isfinite(static_cast<double>(xstart)) && std::isfinite(static_cast<double>(xend)),
                "unsupported range: ",
                xstart,
                " -> ",
                xend);
    TORCH_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
                "upper bound and larger bound inconsistent with step sign");

    TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
                "invalid size, possible overflow?");
    int64_t size = static_cast<int64_t>(size_d);
    int64_t numel = result.numel();

    if (numel != size) {
      if (numel > 0) {
        TORCH_WARN("The number of elements in the out tensor of shape ",
                   result.sizes(),
                   " is ",
                   numel,
                   " which does not match the computed number of elements ",
                   size,
                   ". Note that this may occur as a result of rounding error. "
                   "The out tensor will be resized to a tensor of shape (",
                   size,
                   ",).");
      }
      result.resize_({size});
    }

    if (result.numel() == 0) {
      return;
    }

    bool is_contiguous = result.is_contiguous();
    Tensor r = !is_contiguous ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;
    using namespace mps;
    auto cache_ = MPSGraphCache::getInstance();
    auto stream = getCurrentMPSStream();
    auto mpsDataType = getMPSDataType(result);
    @autoreleasepool {
      string key = "arange_mps_out" + getTensorsStringKey({result}) + ":" + to_string(size);
      auto cachedGraph = cache_->LookUpAs<RangeCachedGraph>(key);
      if (!cachedGraph) {
        cachedGraph = cache_->CreateCachedGraphAs<RangeCachedGraph>(key, ^MPSCachedGraph*() {
          auto mpsGraph = make_mps_graph();
          return new RangeCachedGraph(mpsGraph, mpsDataType, size);
        });
      }
      Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, r);
      NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
      MPSScalar startScalar = getMPSScalar(start, result.scalar_type());
      feeds[cachedGraph->startTensor] = getMPSGraphTensorFromScalar(stream, startScalar);
      MPSScalar stepScalar = getMPSScalar(step, result.scalar_type());
      feeds[cachedGraph->multiplyTensor] = getMPSGraphTensorFromScalar(stream, stepScalar);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
          @{outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()};
      runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    }

    if (!is_contiguous) {
      result.copy_(r);
    }
  });

  return result;
}

Tensor& range_mps_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
  AT_DISPATCH_MPS_TYPES(result.scalar_type(), "arange_mps", [&]() {
    using accscalar_t = at::acc_type_device<scalar_t, kMPS>;
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    // double size_d = ((xend - xstart) / xstep) + 1;
    double size_d;
    if constexpr (std::is_same_v<scalar_t, int64_t>) {
      size_d = static_cast<double>(end.to<accscalar_t>() - start.to<accscalar_t>()) / step.to<accscalar_t>() + 1;
    } else {
      size_d = static_cast<double>(end.to<double>() - start.to<double>()) / step.to<double>() + 1;
    }

    TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
    TORCH_CHECK(std::isfinite(static_cast<double>(xstart)) && std::isfinite(static_cast<double>(xend)),
                "unsupported range: ",
                xstart,
                " -> ",
                xend);
    TORCH_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
                "upper bound and larger bound inconsistent with step sign");

    TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
                "invalid size, possible overflow?");

    int64_t size = static_cast<int64_t>(size_d);

    int64_t numel = result.numel();

    if (numel != size) {
      result.resize_({size});
    }
    bool is_contiguous = result.is_contiguous();
    Tensor r = !is_contiguous ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;
    using namespace mps;
    auto cache_ = MPSGraphCache::getInstance();
    auto stream = getCurrentMPSStream();
    auto mpsDataType = getMPSDataType(result);
    @autoreleasepool {
      string key = "arange_mps_out" + getTensorsStringKey({result}) + ":" + to_string(size);
      auto cachedGraph = cache_->LookUpAs<RangeCachedGraph>(key);
      if (!cachedGraph) {
        cachedGraph = cache_->CreateCachedGraphAs<RangeCachedGraph>(key, ^MPSCachedGraph*() {
          auto mpsGraph = make_mps_graph();
          return new RangeCachedGraph(mpsGraph, mpsDataType, size);
        });
      }
      Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, r);
      NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
      MPSScalar startScalar = getMPSScalar(start, result.scalar_type());
      feeds[cachedGraph->startTensor] = getMPSGraphTensorFromScalar(stream, startScalar);
      MPSScalar stepScalar = getMPSScalar(step, result.scalar_type());
      feeds[cachedGraph->multiplyTensor] = getMPSGraphTensorFromScalar(stream, stepScalar);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
          @{outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()};
      runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    }

    if (!is_contiguous) {
      result.copy_(r);
    }
  });

  return result;
}

Tensor& linspace_out_mps(const Scalar& start, const Scalar& end, int64_t steps, Tensor& result) {
  using namespace mps;

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
      string key =
          "linspace_out_mps:" + getTensorsStringKey({result}) + ":" + to_string(steps) + to_string(start_less_end);
      auto cachedGraph = cache_->LookUpAs<RangeCachedGraph>(key);

      if (!cachedGraph) {
        cachedGraph = cache_->CreateCachedGraphAs<RangeCachedGraph>(key, ^MPSCachedGraph*() {
          RangeCachedGraph* newCachedGraph = nil;

          @autoreleasepool {
            MPSGraph* mpsGraph = make_mps_graph();
            newCachedGraph = new RangeCachedGraph(mpsGraph, MPSDataTypeFloat32, steps, true, start_less_end);

            if (getMPSDataType(result) != MPSDataTypeFloat32) {
              newCachedGraph->outputTensor = [mpsGraph castTensor:newCachedGraph->outputTensor
                                                           toType:getMPSDataType(result)
                                                             name:@"output"];
            }
          }
          return newCachedGraph;
        });
      }

      NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
      auto multiply = (end.to<double>() - start.to<double>()) / ((double)steps - 1.0f);
      Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, r);

      // Create dictionary of inputs and outputs
      MPSScalar startScalar = getMPSScalar(start, ScalarType::Float);
      feeds[cachedGraph->startTensor] = getMPSGraphTensorFromScalar(stream, startScalar);
      MPSScalar endScalar = getMPSScalar(end, ScalarType::Float);
      feeds[cachedGraph->endTensor] = getMPSGraphTensorFromScalar(stream, endScalar);
      MPSScalar multiplyScalar = getMPSScalar(multiply, ScalarType::Float);
      feeds[cachedGraph->multiplyTensor] = getMPSGraphTensorFromScalar(stream, multiplyScalar);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
          @{outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()};
      runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    }

    if (!result.is_contiguous()) {
      result.copy_(r);
    }
  }
  return result;
}

} // namespace at::native
