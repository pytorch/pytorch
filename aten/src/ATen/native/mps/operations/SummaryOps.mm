//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/Resize.h>

namespace at::native {
namespace mps {

static const char* METAL_SUMMARY = R"SUMMARY_METAL(

#include <metal_stdlib>
using namespace metal;

#define REGISTER_CROSS_FUNC(DTYPE)                              \
static inline DTYPE ## 3 cross(DTYPE ## 3 x, DTYPE ## 3 y) {    \
  DTYPE ## 3 out;                                               \
  out.x = x.y * y.z - x.z * y.y;                                \
  out.y = x.z * y.x - x.x * y.z;                                \
  out.z = x.x * y.y - x.y * y.x;                                \
  return out;                                                   \
}

// Metal only supports half and float for native cross implementation.
// For all the the other data types, implement cross manually.
REGISTER_CROSS_FUNC(int);
REGISTER_CROSS_FUNC(long);
REGISTER_CROSS_FUNC(short);
REGISTER_CROSS_FUNC(char);
REGISTER_CROSS_FUNC(uchar);
REGISTER_CROSS_FUNC(bool);

template<typename T, typename U>
kernel void cross(constant void     * input_        [[buffer(0)]],
                  constant void     * other_        [[buffer(1)]],
                  device   void     * out_          [[buffer(2)]],
                  constant uint3    * offsets       [[buffer(3)]],
                  constant int64_t  & outStride     [[buffer(4)]],
                  constant int64_t  & inputStride   [[buffer(5)]],
                  constant int64_t  & otherStride   [[buffer(6)]],
                  uint tid [[thread_position_in_grid]]) {
  device   T* out   = (device   T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);

  const U x = {input[0 * inputStride], input[1 * inputStride], input[2 * inputStride]};
  const U y = {other[0 * otherStride], other[1 * otherStride], other[2 * otherStride]};
  const U res = cross(x, y);

  out[0 * outStride] = res.x;
  out[1 * outStride] = res.y;
  out[2 * outStride] = res.z;
}

#define REGISTER_CROSS_OP(DTYPE)                       \
template                                               \
[[host_name("cross_" #DTYPE)]]                         \
kernel void cross<DTYPE, DTYPE ## 3>(                  \
  constant void     * input_        [[buffer(0)]],     \
  constant void     * other_        [[buffer(1)]],     \
  device   void     * out_          [[buffer(2)]],     \
  constant uint3    * offsets       [[buffer(3)]],     \
  constant int64_t  & outStride     [[buffer(4)]],     \
  constant int64_t  & inputStride   [[buffer(5)]],     \
  constant int64_t  & otherStride   [[buffer(6)]],     \
  uint tid [[thread_position_in_grid]]);

REGISTER_CROSS_OP(float);
REGISTER_CROSS_OP(half);
REGISTER_CROSS_OP(int);
REGISTER_CROSS_OP(long);
REGISTER_CROSS_OP(short);
REGISTER_CROSS_OP(char);
REGISTER_CROSS_OP(uchar);
REGISTER_CROSS_OP(bool);

)CROSS_METAL";

static id<MTLLibrary> compileSummaryOpLibrary(id<MTLDevice> device) {
  static id<MTLLibrary> summaryLibrary = nil;
  if (summaryLibrary) {
    return summaryLibrary;
  }

  NSError *error = nil;
  MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion: MTLLanguageVersion2_3];
  summaryLibrary  = [device newLibraryWithSource:[NSString stringWithCString: METAL_SUMMARY encoding:NSASCIIStringEncoding]
                                       options:options
                                         error:&error];
  TORCH_CHECK(summaryLibrary, "Failed to create metal summary library, error: ", [[error description] UTF8String]);
  return summaryLibrary;
}

static id<MTLComputePipelineState> crossPipelineState(id<MTLDevice> device, ScalarType scalar_type) {
  std::string kernel = "cross_" + scalarToMetalTypeString(scalar_type);
  static std::unordered_map<std::string, id<MTLComputePipelineState>> psoCache;
  id<MTLComputePipelineState> pso = psoCache[kernel];
  if (pso) {
    return pso;
  }

  NSError* error = nil;
  id<MTLLibrary> crossLib = compileSummaryOpLibrary(device);
  id<MTLFunction> crossFunc = [crossLib newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
  TORCH_CHECK(crossFunc, "Failed to create function state object for: ", kernel);
  pso = [device newComputePipelineStateWithFunction:crossFunc error:&error];
  TORCH_CHECK(pso, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

  psoCache[kernel] = pso;
  return pso;
}

void histogramdd_impl(
    Tensor& hist_output,
    const TensorList& bin_edges,
    const Tensor& input,
    const c10::optional<Tensor>& weight) {
  TORCH_CHECK(input.dtype() != at::kDouble, "float64 is not supported on MPS");
  TORCH_INTERNAL_ASSERT(input.dim() == 2);

  const int64_t N = input.size(0);
  if (weight.has_value()) {
      TORCH_INTERNAL_ASSERT(weight.value().dim() == 1 && weight.value().numel() == N);
  }

  const int64_t D = input.size(1);
  TORCH_INTERNAL_ASSERT(int64_t(bin_edges.size()) == D);
  for (const auto dim : c10::irange(D)) {
      TORCH_INTERNAL_ASSERT(bin_edges[dim].is_contiguous());
      TORCH_INTERNAL_ASSERT(hist.size(dim) + 1 == bin_edges[dim].numel());
  }

  if (D == 0) {
      // hist is an empty tensor in this case; nothing to do here
      return;
  }

  auto iter = TensorIteratorConfig()
      .add_output(out)
      .add_input(input)
      .add_input(other)
      .resize_outputs(false)
      .declare_static_shape(out.sizes(), /*squash_dims=*/dim)
      .build();

  id<MTLBuffer> inputBuffer  = getMTLBufferStorage(input);
  id<MTLBuffer> otherBuffer  = getMTLBufferStorage(other);
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(out);
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  const int64_t out_dim_stride =  out.stride(dim);
  const int64_t input_dim_stride = input.stride(dim);
  const int64_t other_dim_stride = other.stride(dim);
  const uint32_t nDim = iter.ndim();
  constexpr uint32_t nOffsets = 3;
  const uint32_t numThreads = iter.numel();
  dispatch_sync(mpsStream->queue(), ^(){
    @autoreleasepool {
      NSError* error = nil;
      id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      const IntArrayRef& iterShape = iter.shape();
      std::vector<uint32_t> iterShapeData(iterShape.size());
      std::vector<std::array<uint32_t, nOffsets>> strides(nDim);

      for (const auto i: c10::irange(iterShape.size())) {
        TORCH_CHECK(i <= UINT32_MAX);
        iterShapeData[i] = (uint32_t)(iterShape[i]);
      }

      for (const auto i: c10::irange(nDim)) {
        for (const auto offset: c10::irange(nOffsets)) {
            strides[i][offset] = iter.strides(offset)[i];
        }
      }

      id<MTLFunction> kernelDataOffsetsFunction = MPSDevice::getInstance()->metalIndexingFunction("kernel_index_offsets", nil);
      id<MTLComputePipelineState> kernelDataOffsetsPSO = [[device newComputePipelineStateWithFunction: kernelDataOffsetsFunction
                                                                                                error: &error] autorelease];
      id<MTLBuffer> kernelDataOffsets = [[device newBufferWithLength: numThreads * sizeof(simd_uint3)
                                                             options: 0] autorelease];
      TORCH_CHECK(kernelDataOffsetsPSO, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);
      [computeEncoder setComputePipelineState:kernelDataOffsetsPSO];
      [computeEncoder setBytes:strides.data() length:sizeof(uint32_t) * nDim * nOffsets atIndex:0];
      [computeEncoder setBuffer:kernelDataOffsets offset:0 atIndex:1];
      [computeEncoder setBytes:iterShapeData.data() length:sizeof(uint32_t) * iterShape.size() atIndex:2];
      [computeEncoder setBytes:&nDim length:sizeof(uint32_t) atIndex:3];
      [computeEncoder setBytes:&nOffsets length:sizeof(uint32_t) atIndex:4];

      NSUInteger kernelOffsetsTGSize = kernelDataOffsetsPSO.maxTotalThreadsPerThreadgroup;
      if (kernelOffsetsTGSize > numThreads)
          kernelOffsetsTGSize = numThreads;

      MTLSize kernelOffsetsThreadGroupSize = MTLSizeMake(kernelOffsetsTGSize, 1, 1);
      [computeEncoder dispatchThreads: gridSize
                threadsPerThreadgroup: kernelOffsetsThreadGroupSize];

      id<MTLComputePipelineState> crossPSO = crossPipelineState(device, out.scalar_type());
      [computeEncoder setComputePipelineState:crossPSO];
      [computeEncoder setBuffer:inputBuffer  offset:input.storage_offset() * input.element_size() atIndex:0];
      [computeEncoder setBuffer:otherBuffer  offset:other.storage_offset() * other.element_size() atIndex:1];
      [computeEncoder setBuffer:outputBuffer offset:out.storage_offset() * out.element_size() atIndex:2];
      [computeEncoder setBuffer:kernelDataOffsets offset:0 atIndex:3];
      [computeEncoder setBytes:&out_dim_stride  length:sizeof(int64_t)  atIndex:4];
      [computeEncoder setBytes:&input_dim_stride length:sizeof(int64_t) atIndex:5];
      [computeEncoder setBytes:&other_dim_stride length:sizeof(int64_t) atIndex:6];

      NSUInteger tgSize = crossPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > numThreads) {
          tgSize = numThreads;
      }

      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreads: gridSize
                threadsPerThreadgroup: threadGroupSize];

      [computeEncoder endEncoding];
      mpsStream->commit(true);
    }
  });
}

void bincount_histc_mps_impl(
    Tensor& self,
    const Tensor& weights,
    const Tensor& output) 
  {
  using namespace mps;

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* weightsTensor_ = nil;
    MPSGraphTensor* scatterDataTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  bool has_weights = weights.defined();

  @autoreleasepool {
    string key = "bincount_mps_impl" + getTensorsStringKey({self, weights});
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          // Initialize graph
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);
          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor *scatterDataTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSScalarType(output.scalar_type()));

          MPSGraphTensor *updatesTensor = nil;
          if (has_weights) {
            updatesTensor = mpsGraphRankedPlaceHolder(mpsGraph, weights);
          }
          else {
            updatesTensor = [mpsGraph constantWithScalar:1.0f
                                                   shape:getMPSShape(self)
                                                dataType:getMPSDataType(output.scalar_type())];
          }

          MPSGraphTensor *castedInputTensor = inputTensor;
          if (self.scalar_type() == kByte) {
            castedInputTensor = [mpsGraph castTensor:inputTensor
                                              toType:MPSDataTypeInt32
                                                name:@"castInputTensor"];
          }

          MPSGraphTensor *outputTensor = [mpsGraph scatterWithDataTensor:scatterDataTensor
                                                           updatesTensor:updatesTensor
                                                           indicesTensor:castedInputTensor
                                                                     axis:0
                                                                     mode:MPSGraphScatterModeAdd
                                                                     name:nil];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
          newCachedGraph->scatterDataTensor_ = scatterDataTensor;
          if (has_weights) {
            newCachedGraph->weightsTensor_ = updatesTensor;
          }
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    // Create placeholders which use the keys of the CachedGraph to create inputs and outputs of the operation
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    Placeholder scatterPlaceholder = Placeholder(cachedGraph->scatterDataTensor_, output);
    Placeholder weightsPlaceholder = Placeholder();

    // Create dictionary of inputs/feeds and outputs/results
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =[NSMutableDictionary dictionary];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[scatterPlaceholder.getMPSGraphTensor()] = scatterPlaceholder.getMPSGraphTensorData();
    if(has_weights) {
      weightsPlaceholder = Placeholder(cachedGraph->weightsTensor_, weights);
      feeds[weightsPlaceholder.getMPSGraphTensor()] = weightsPlaceholder.getMPSGraphTensorData();
    }

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    // Run the graph
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}

Tensor _bincount_histc_mps(
    const Tensor& self,
    const c10::optional<Tensor>& weights_opt,
    const c10::optional<int64_t> minlength_opt,
    const c10::optional<int64_t> nbins_opt,
    const c10::optional<Scalar>& min_opt,
    const c10::optional<Scalar>& max_opt,
    const c10::optional<Tensor>& output_opt)
  {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weights_maybe_owned = at::borrow_from_optional_tensor(weights_opt);
  const Tensor& weights = *weights_maybe_owned;
  Tensor weights_ = weights;
  bool has_weights = weights.defined();

  Tensor self_ = self;
  int64_t nbins;
  int64_t minlength = 0;

  if (nbins_opt && min_opt && max_opt) {  // histc
    nbins = nbins_opt.value();
    at::li
    
    if (min_opt.value().toSymInt() != 0 || max_opt.value().toSymInt() != 0){
      auto indices = at::masked_select(self, at::where(at::logical_and(self > min_opt.value(), self < max_opt.value()), 1, 0));
      self_ = at::argsort(indices);
    } else {
      self_ = at::argsort(self);
    }
    
  } else {  // bincount
    TORCH_CHECK(c10::isIntegralType(self.scalar_type(), /*includesBool=*/true));
    TORCH_CHECK(self.dim() == 1 && self.min().item<int64_t>() >= 0, "bincount only supports 1-d non-negative integral inputs.");
    TORCH_CHECK(!(has_weights && (weights.dim() != 1 || weights.size(0) != self.size(0))), "weights should be 1-d and have the same length as input");
    
    if (minlength_opt) {
      minlength = minlength_opt.value();
      TORCH_CHECK(minlength >= 0, "minlength should be >= 0");
    }

    if (self.dim() == 1 && self.numel() == 0) {
      return at::zeros(
          {minlength},
          kLong,
          c10::nullopt /* layout */,
          kMPS,
          c10::nullopt /* pin_memory */);
    }
    nbins = std::max(self.max().item<int64_t>() + 1L, minlength);
  }

  if (output_opt) {
    auto output_maybe_owned = at::borrow_from_optional_tensor(output_opt);
    auto output = *output_maybe_owned;
    resize_output(output, IntArrayRef({nbins}));
    bincount_histc_mps_impl(self_, weights_, output);
    return output;
  } else {
    Tensor output;

    if (has_weights) {
      if(weights.scalar_type() != ScalarType::Float &&
        weights.scalar_type() != ScalarType::Int   &&
        weights.scalar_type() != ScalarType::Half) {
          // Scatter doesn't work for int8/int16 dtypes
          weights_ = weights.to(kInt);
      }
      output = at::zeros(
          {nbins},
          optTypeMetaToScalarType(weights_.options().dtype_opt()),
          weights_.options().layout_opt(),
          weights_.options().device_opt(),
          weights_.options().pinned_memory_opt());
    } else {
      output = at::zeros(
          {nbins},
          (nbins_opt && min_opt && max_opt) ? self.scalar_type() : kLong,
          c10::nullopt /* layout */,
          kMPS,
          c10::nullopt /* pin_memory */);
    }
    bincount_histc_mps_impl(self_, weights_, output);
    return output;
  }
}
} // namespace mps

Tensor _bincount_mps(const Tensor& self, const c10::optional<Tensor>& weights_opt, int64_t minlength) {
  return mps::_bincount_histc_mps(
    self,
    weights_opt,
    minlength,
    c10::nullopt /* nbins_opt */,
    c10::nullopt /* min_opt */,
    c10::nullopt /* max_opt */,
    c10::nullopt /* output_opt */
  );
}

Tensor _histc_mps(
    const Tensor& self,
    int64_t nbins,
    const Scalar& min,
    const Scalar& max) {
  return mps::_bincount_histc_mps(
    self,
    c10::nullopt /* weights_opt */,
    c10::nullopt /* minlength_opt */,
    c10::optional<int64_t>(nbins),
    c10::optional<Scalar>(min),
    c10::optional<Scalar>(max),
    c10::nullopt /* output_opt */
  );
}
Tensor& _histc_out_mps(const Tensor& self, int64_t bins, const Scalar& min, const Scalar& max, Tensor& result) {
  mps::_bincount_histc_mps(
    self,
    c10::nullopt /* weights_opt */,
    c10::nullopt /* minlength_opt */,
    c10::optional<int64_t>(bins),
    c10::optional<Scalar>(min),
    c10::optional<Scalar>(max),
    c10::optional<Tensor>(result)
  );
  return result;
}

} // namespace at::native
