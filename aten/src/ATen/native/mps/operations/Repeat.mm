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
    std::string key = "repeat_mps:" + getTensorsStringKey(self) + ":" + getArrayRefString(repeats);
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

Tensor repeat_interleave_mps(const Tensor& repeat, std::optional<int64_t> output_size) {
  TORCH_CHECK(repeat.dim() == 1, "repeat_interleave only accept 1D vector as repeat");
  std::string scalar_type;
  if (repeat.scalar_type() == kInt) {
    scalar_type = "int32_t";
  } else if (repeat.scalar_type() == kLong) {
    scalar_type = "int64_t";
  } else {
    TORCH_CHECK(false, "repeats has to be Long or Int tensor");
  }
  if (repeat.size(0) == 0) {
    return at::empty_like(repeat, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  Tensor repeat_ = repeat.contiguous();
  Tensor cumsum = repeat.cumsum(0);
  int64_t total = 0;
  if (output_size.has_value()) {
    total = output_size.value();
  } else {
    total = cumsum[-1].item<int64_t>();
    TORCH_CHECK((repeat >= 0).all().item<uint8_t>(), "repeats can not be negative");
  }

  auto result = at::empty({total}, repeat.options());

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      auto pipelineState = lib.getPipelineStateForFunc(fmt::format("repeat_interleave_{}", scalar_type));

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(pipelineState, "repeat_interleave:" + scalar_type, false);

      [computeEncoder setComputePipelineState:pipelineState];
      mps::mtl_setArgs(computeEncoder, repeat_, cumsum, result, repeat.size(0));
      mps::mtl_dispatch1DJob(computeEncoder, pipelineState, repeat.size(0));

      getMPSProfiler().endProfileKernel(pipelineState);
    }
  });
  return result;
}

} // namespace at::native
