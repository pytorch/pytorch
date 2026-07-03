//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/RangeUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/linspace_native.h>
#include <ATen/ops/range_native.h>
#include <cmath>
#include <limits>

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/RangeFactories_metallib.h>
#endif

namespace {
// Mirror the kernel-side layouts in RangeFactories.metal.
struct RangeVals {
  float start;
  float step;
  float end;
};
template <typename I>
struct RangeIdx {
  I halfway;
  I steps;
  I stride;
};

struct RangeCachedGraph : public mps::MPSCachedGraph {
  API_AVAILABLE(macosx(12.3))
  RangeCachedGraph(MPSGraph* mpsGraph, MPSDataType dataType, int32_t shapeVal) : MPSCachedGraph(mpsGraph) {
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
    }
  }
  MPSGraphTensor* startTensor = nil;
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
      TORCH_CHECK_VALUE(xstep != 0, "step must be nonzero");
      size_d = std::ceil(static_cast<double>(end.to<accscalar_t>() - start.to<accscalar_t>()) / step.to<accscalar_t>());
    } else {
      size_d = std::ceil(static_cast<double>(end.to<double>() - start.to<double>()) / step.to<double>());
    }

    arange_check_bounds(start, end, step);

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

    bool needs_gather = !mps::needsGather(result);
    Tensor r = !needs_gather ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;
    using namespace mps;
    auto cache_ = MPSGraphCache::getInstance();
    auto stream = getCurrentMPSStream();
    auto mpsDataType = getMPSDataType(result);
    @autoreleasepool {
      std::string key = "arange_mps_out" + getTensorsStringKey({result}) + ":" + std::to_string(size);
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

      runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
    }

    if (!needs_gather) {
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

    arange_check_bounds(start, end, step);

    TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
                "invalid size, possible overflow?");

    int64_t size = static_cast<int64_t>(size_d);

    int64_t numel = result.numel();

    if (numel != size) {
      result.resize_({size});
    }
    bool needs_gather = !mps::needsGather(result);
    Tensor r = !needs_gather ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;
    using namespace mps;
    auto cache_ = MPSGraphCache::getInstance();
    auto stream = getCurrentMPSStream();
    auto mpsDataType = getMPSDataType(result);
    @autoreleasepool {
      std::string key = "arange_mps_out" + getTensorsStringKey({result}) + ":" + std::to_string(size);
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

      runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
    }

    if (!needs_gather) {
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
    return result;
  }
  if (steps == 1) {
    result.fill_(start);
    return result;
  }

  // Integral outputs truncate the endpoints to the integer type first (matches
  // CPU/CUDA); float/complex interpolate the real endpoints in float32.
  RangeVals vals{};
  if (isIntegralType(result.scalar_type(), /*includeBool=*/false)) {
    AT_DISPATCH_INTEGRAL_TYPES(result.scalar_type(), "linspace_mps", [&]() {
      const auto s = static_cast<float>(start.to<scalar_t>());
      const auto e = static_cast<float>(end.to<scalar_t>());
      vals = {s, (e - s) / static_cast<float>(steps - 1), e};
    });
  } else {
    const auto s = start.to<float>();
    const auto e = end.to<float>();
    vals = {s, (e - s) / static_cast<float>(steps - 1), e};
  }
  const auto halfway = steps / 2;

  auto stream = getCurrentMPSStream();
  auto encoder = stream->commandEncoder();
  const auto tname = scalarToMetalTypeString(result);

  if (result.is_contiguous() || result.dim() == 1) {
    // Fast path: contiguous (stride 1) or a 1-D strided view writes in place at
    // index*stride, no scatter. Narrow the index math to int32 when it fits.
    const auto stride = result.is_contiguous() ? 1 : result.stride(0);
    const auto abs_stride = stride < 0 ? -stride : stride;
    const auto use32 = std::max<int64_t>(steps, (steps - 1) * abs_stride) <= std::numeric_limits<int32_t>::max();
    auto pso = lib.getPipelineStateForFunc("linspace_" + tname + (use32 ? "_i32" : "_i64"));
    dispatch_sync(stream->queue(), ^() {
      @autoreleasepool {
        [encoder setComputePipelineState:pso];
        if (use32) {
          RangeIdx<int32_t> idx{int32_t(halfway), int32_t(steps), int32_t(stride)};
          mtl_setArgs(encoder, result, vals, idx);
        } else {
          RangeIdx<int64_t> idx{halfway, steps, stride};
          mtl_setArgs(encoder, result, vals, idx);
        }
        mtl_dispatch1DJob(encoder, pso, static_cast<NSUInteger>(steps));
      }
    });
  } else {
    // Multi-dim non-contiguous out=: scatter to each element's strided offset.
    auto pso = lib.getPipelineStateForFunc("linspace_strided_" + tname);
    const auto ndim = static_cast<int>(result.dim());
    const auto sizes = result.sizes();
    const auto strides = result.strides();
    RangeIdx<int64_t> idx{halfway, steps, 1};
    dispatch_sync(stream->queue(), ^() {
      @autoreleasepool {
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, result, vals, idx, ndim);
        [encoder setBytes:sizes.data() length:ndim * sizeof(int64_t) atIndex:4];
        [encoder setBytes:strides.data() length:ndim * sizeof(int64_t) atIndex:5];
        mtl_dispatch1DJob(encoder, pso, static_cast<NSUInteger>(steps));
      }
    });
  }
  return result;
}

} // namespace at::native
