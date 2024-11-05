//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Repeat.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/permute_native.h>
#include <ATen/ops/repeat_interleave_native.h>
#include <ATen/ops/repeat_native.h>
#include <fmt/format.h>

namespace at::native {

Tensor permute_mps(const Tensor& self, IntArrayRef dims) {
  auto nDims = self.dim();
  TORCH_CHECK(dims.size() == (size_t)nDims, "number of dims don't match in permute");
  auto oldSizes = self.sizes();
  auto oldStrides = self.strides();
  DimVector newSizes(nDims);
  DimVector newStrides(nDims);
  std::vector<bool> seen(nDims);
  for (const auto i : c10::irange(nDims)) {
    auto dim = maybe_wrap_dim(dims[i], nDims);
    TORCH_CHECK(!seen[dim], "repeated dim in permute");
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    newStrides[i] = oldStrides[dim];
  }
  return self.as_strided(newSizes, newStrides);
}

Tensor repeat_mps(const Tensor& self, IntArrayRef repeats) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  TORCH_CHECK(repeats.size() >= (size_t)self.dim(),
              "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");
  TORCH_CHECK(!self.is_complex(), "repeat(): Not supported for complex yet!");

  // Add new leading dimensions to the tensor if the
  // number of target dimensions is larger than the
  // number of source dimensions.
  int64_t num_new_dimensions = repeats.size() - self.dim();
  DimVector padded_size(num_new_dimensions, 1);
  padded_size.insert(padded_size.end(), self.sizes().begin(), self.sizes().end());
  DimVector target_size(repeats.size());
  bool zero_tensor = false;
  for (const auto idx : c10::irange(repeats.size())) {
    if (repeats[idx] == 0) {
      zero_tensor = true;
    }
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  Tensor expanded_tensor = self.expand(padded_size);
  Tensor result = at::empty(target_size, self.options());
  if (zero_tensor || result.numel() == 0) {
    return result;
  }

  auto stream = at::mps::getCurrentMPSStream();
  auto inputDataType = getMPSDataType(expanded_tensor);
  auto outputDataType = getMPSDataType(result);

  @autoreleasepool {
    string key = "repeat_mps:" + getTensorsStringKey(self) + ":" + getArrayRefString(repeats);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, inputDataType, getMPSShape(expanded_tensor));
      MPSGraphTensor* outputTensor = [mpsGraph tileTensor:inputTensor withMultiplier:getMPSShape(repeats) name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(
        cachedGraph->inputTensor_, expanded_tensor, /*mpsShape=*/nil, /*gatherTensorData=*/true, inputDataType);
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor_, result, /*mpsShape=*/nil, /*gatherTensorData*/ false, outputDataType);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return result;
}

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Repeat_metallib.h>
#endif

template <typename index_t>
void computeRepeatIndices(const index_t* repeat_ptr,
                          const int64_t* cumsum_ptr,
                          index_t* result_ptr,
                          int64_t size,
                          int64_t result_size) {
  id<MTLBuffer> repeatBuffer = reinterpret_cast<id<MTLBuffer>>(repeat_ptr);
  id<MTLBuffer> cumsumBuffer = reinterpret_cast<id<MTLBuffer>>(cumsum_ptr);
  id<MTLBuffer> resultBuffer = reinterpret_cast<id<MTLBuffer>>(result_ptr);
  TORCH_CHECK(repeatBuffer && cumsumBuffer && resultBuffer);

  std::string scalar_type;
  if constexpr (std::is_same_v<index_t, int32_t>) {
    scalar_type = "int32_t";
  } else if constexpr (std::is_same_v<index_t, int64_t>) {
    scalar_type = "int64_t";
  } else {
    TORCH_CHECK(false, "repeat_interleave: unsupported indexing data type");
  }

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      auto pipelineState = lib.getPipelineStateForFunc(fmt::format("repeat_interleave_{}", scalar_type));

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(pipelineState, "repeat_interleave:" + scalar_type, false);

      [computeEncoder setComputePipelineState:pipelineState];
      [computeEncoder setBuffer:repeatBuffer offset:0 atIndex:0];
      [computeEncoder setBuffer:cumsumBuffer offset:0 atIndex:1];
      [computeEncoder setBuffer:resultBuffer offset:0 atIndex:2];
      mps::mtl_setBytes(computeEncoder, size, 3);
      mps::mtl_dispatch1DJob(computeEncoder, pipelineState, size);

      getMPSProfiler().endProfileKernel(pipelineState);
    }
  });
}

Tensor repeat_interleave_mps(const Tensor& repeat_, std::optional<int64_t> output_size) {
  Tensor output;
  Tensor repeat = repeat_;
  if (repeat.scalar_type() == kLong && !is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS)) {
    // #103810551: `repeat_interleave_common` uses cumsum to calculate the final shape of output,
    // which currently doesn't support int64_t as input. Casting internally the indices to int32_t.
    TORCH_WARN_ONCE(
        "MPS: no support for int64 repeats mask, casting it to int32. Support has been added in macOS 13.3");
    repeat = repeat.to(kInt);
  }
  AT_DISPATCH_INDEX_TYPES(repeat.scalar_type(), "repeat_interleave_mps", [&]() {
    output = repeat_interleave_common<index_t, computeRepeatIndices<index_t>>(repeat, output_size);
  });
  return output;
}

} // namespace at::native
