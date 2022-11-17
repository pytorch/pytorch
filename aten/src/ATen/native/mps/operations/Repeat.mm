//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Repeat.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/repeat_interleave_native.h>
#endif

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

template <typename index_t>
static void compute_mps_kernel(index_t *repeat_ptr,
    	int64_t *cumsum_ptr,
    	index_t *result_ptr,
    	int64_t size,
    	int64_t result_size)
    {
    	using namespace mps;

    	id<MTLBuffer> inputBuffer = getMTLBufferStorage(inputTensor);
    	id<MTLBuffer> outputBuffer = getMTLBufferStorage(outputTensor);
    	MPSStream *mpsStream = getCurrentMPSStream();
    	id<MTLDevice> device = MPSDevice::getInstance()->device();

    	dispatch_sync(mpsStream->queue(), ^ ()
    		{         
        @autoreleasepool {

    				NSError *error = nil;
    				TORCH_CHECK(result_size == cumsum_ptr[size - 1], "allocated size does not match required size");
    				int64_t idx = blockIdx.x *blockDim.x + threadIdx.x;
    				int64_t stride = (blockDim.x *gridDim.x) / at::mps::C10_WARP_SIZE;
    				int warp_id = idx / at::mps::C10_WARP_SIZE;
    				int tid_in_warp = idx % at::mps::C10_WARP_SIZE;
    				for (int64_t i = warp_id; i < size; i += stride)
    				{
    					int64_t end = cumsum_ptr[i];
    					index_t repeat = repeat_ptr[i];
    					TORCH_CHECK(repeat >= 0, "repeats can not be negative"; int64_t start = end - repeat);
    						for (int64_t j = start + tid_in_warp; j < end; j += at::mps::C10_WARP_SIZE)
    						{
    							result_ptr[j] = i;
    						}
    					}

    				int64_t block = 512;
    				int64_t warps_per_block = block / at::mps::C10_WARP_SIZE();

    				NSUInteger threadGroupSize = ((size + warps_per_block - 1) / warps_per_block, 2048);
    				if (threadGroupSize > 2048)
    				{
    					threadGroupSize = 2048;
    				}

    				MTLSize grid = MTLSizeMake(threadGroupSize, 1, 1);

    			[computeEncoder dispatchThreads: gridSize threadsPerThreadgroup: grid];

    			[computeEncoder endEncoding];
    				mpsStream->commit(true);

        }        
      });
      
    return true;
  }

template <typename index_t>
void compute_mps(index_t* repeat_ptr,
    int64_t* cumsum_ptr,
    index_t* result_ptr,
    int64_t size,
    int64_t result_size) {

  TORCH_CHECK(compute_mps_kernel(
      repeat_ptr, cumsum_ptr, result_ptr, size, result_size), "Failed compute_mps");
}


namespace at {
namespace native {

Tensor permute_mps(const Tensor& self, IntArrayRef dims) {
  auto nDims = self.dim();
  TORCH_CHECK(dims.size() == (size_t)nDims,
           "number of dims don't match in permute");
  auto oldSizes = self.sizes();
  auto oldStrides = self.strides();
  DimVector newSizes(nDims);
  DimVector newStrides(nDims);
  std::vector<bool> seen(nDims);
  for (const auto i : c10::irange(nDims)) {
    auto dim = maybe_wrap_dim(dims[i], nDims);
    TORCH_CHECK(!seen[dim],
             "repeated dim in permute");
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    newStrides[i] = oldStrides[dim];
  }
  return self.as_strided(newSizes, newStrides);
}

Tensor repeat_mps(const Tensor& self, IntArrayRef repeats) {

  using namespace mps;

  TORCH_CHECK(repeats.size() >= (size_t)self.dim(),
           "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  // Add new leading dimensions to the tensor if the
  // number of target dimensions is larger than the
  // number of source dimensions.
  int64_t num_new_dimensions = repeats.size() - self.dim();
  DimVector padded_size(num_new_dimensions, 1);
  padded_size.insert(padded_size.end(), self.sizes().begin(), self.sizes().end());
  DimVector target_size(repeats.size());
  bool zero_tensor = false;
  for(const auto idx : c10::irange(repeats.size())) {
    if (repeats[idx] == 0) {
      zero_tensor = true;
    }
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  Tensor expanded_tensor = self.expand(padded_size);
  Tensor result = at::empty(target_size, self.options());
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  if(zero_tensor || result.numel() == 0) {
    return result;
  }

  auto stream = at::mps::getCurrentMPSStream();
  auto inputDataType = getMPSDataType(expanded_tensor.scalar_type());
  auto outputDataType = getMPSDataType(result.scalar_type());
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
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, inputDataType, getMPSShape(expanded_tensor));
          MPSGraphTensor* outputTensor = [mpsGraph tileTensor:inputTensor
                                               withMultiplier:getMPSShape(repeats)
                                                         name:nil];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(
      cachedGraph->inputTensor_, expanded_tensor, /*mpsShape=*/nil, /*gatherTensorData=*/true, inputDataType);
    Placeholder outputPlaceholder = Placeholder(
      cachedGraph->outputTensor_, result, /*mpsShape=*/nil, /*gatherTensorData*/false, outputDataType);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return result;
}

Tensor repeat_interleave_mps(const Tensor& repeats,c10::optional<int64_t> output_size) {
  Tensor output;
  AT_DISPATCH_INDEX_TYPES(
      repeat.scalar_type(), "repeat_interleave_cuda", [&]() {
        output = repeat_interleave_common<index_t, compute_mps<index_t>>(
            repeat, output_size);
      });
  return output;
}

}
}
