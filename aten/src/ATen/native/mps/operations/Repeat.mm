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

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

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
  if (!is_macos_13_or_newer()) {
    if (expanded_tensor.scalar_type() == kBool) {
      inputDataType = MPSDataTypeInt8;
    }
    if (result.scalar_type() == kBool) {
      outputDataType = MPSDataTypeInt8;
    }
  }

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

static const char* METAL_REPEAT_INTERLEAVE = R"METAL_REPEAT(
kernel void repeat_interleave(constant {0}     * repeat_ptr                [[buffer(0)]],
                              constant int64_t * cumsum_ptr                [[buffer(1)]],
                              device {0}       * result_ptr                [[buffer(2)]],
                              uint               threads_per_threadgroup   [[threads_per_threadgroup]],
                              uint               tid                       [[thread_position_in_grid]]) {{
  int64_t end = cumsum_ptr[tid];
  {0} repeat = repeat_ptr[tid];
  int64_t start = end - repeat;
  for (uint j = start; j < end; j++) {{
    result_ptr[j] = tid;
  }}
}}
)METAL_REPEAT";

static id<MTLLibrary> compileRepeatInterleaveLib(id<MTLDevice> device, const std::string& t1) {
  auto key = t1;
  static std::unordered_map<std::string, id<MTLLibrary>> libMap;
  auto it = libMap.find(key);
  if (it != libMap.end()) {
    return it->second;
  }
  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion2_3];
  auto rc =
      [device newLibraryWithSource:[NSString stringWithUTF8String:fmt::format(METAL_REPEAT_INTERLEAVE, t1).c_str()]
                           options:options
                             error:&error];
  TORCH_CHECK(rc != nil && error == nil, "Failed to compile library: ", [[error localizedDescription] UTF8String]);
  libMap[key] = rc;
  return rc;
}

static id<MTLComputePipelineState> getPipelineState(id<MTLDevice> device, const std::string& t1) {
  static std::string kernel = "repeat_interleave";
  auto key = kernel + t1;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cplMap;
  auto it = cplMap.find(key);
  if (it != cplMap.end()) {
    return it->second;
  }
  NSError* error = nil;
  auto library = compileRepeatInterleaveLib(device, t1);
  id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
  TORCH_CHECK(func != nil, "Can't get kernel ", kernel);
  auto rc = [device newComputePipelineStateWithFunction:func error:&error];
  TORCH_CHECK(
      rc != nil && error == nil, "Failed to construct pipeline state: ", [[error localizedDescription] UTF8String]);
  cplMap[key] = rc;
  return rc;
}

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
  if (typeid(index_t) == typeid(int32_t)) {
    scalar_type = "int32_t";
  } else if (typeid(index_t) == typeid(int64_t)) {
    scalar_type = "int64_t";
  } else {
    TORCH_CHECK(false, "repeat_interleave: unsupported indexing data type");
  }

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> pipelineState = getPipelineState(MPSDevice::getInstance()->device(), scalar_type);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(pipelineState, "repeat_interleave:" + scalar_type, false);

      [computeEncoder setComputePipelineState:pipelineState];
      [computeEncoder setBuffer:repeatBuffer offset:0 atIndex:0];
      [computeEncoder setBuffer:cumsumBuffer offset:0 atIndex:1];
      [computeEncoder setBuffer:resultBuffer offset:0 atIndex:2];
      [computeEncoder setBytes:&size length:sizeof(size) atIndex:3];
      MTLSize gridSize = MTLSizeMake(size, 1, 1);
      NSUInteger threadsPerThreadgroup_ = pipelineState.maxTotalThreadsPerThreadgroup;
      if (threadsPerThreadgroup_ > static_cast<NSUInteger>(size)) {
        threadsPerThreadgroup_ = size;
      }
      MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerThreadgroup_, 1, 1);

      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerThreadgroup];

      getMPSProfiler().endProfileKernel(pipelineState);
    }
  });
}

Tensor repeat_interleave_mps(const Tensor& repeat_, c10::optional<int64_t> output_size) {
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
