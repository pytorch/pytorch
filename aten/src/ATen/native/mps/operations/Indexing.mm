//  Copyright © 2022 Apple Inc.
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>

#include <ATen/ceil_div.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/Indexing.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/Resize.h>
#include <ATen/AccumulateType.h>
#include <torch/library.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/IndexingUtils.h>
#include <c10/util/irange.h>
#include <c10/core/QScheme.h>
#include <c10/util/SmallVector.h>
#include <ATen/native/IndexKernel.h>

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace at::native {

static
bool dispatchIndexKernel(TensorIteratorBase& iter,
                         IntArrayRef index_size,
                         IntArrayRef index_stride,
                         bool index_select,
                         bool accumulate) {
  using namespace mps;

 if (iter.numel() == 0)
    return true;

  const Tensor& inputTensor = iter.tensor(1);
  Tensor outputTensor = iter.tensor(0);
  id<MTLBuffer> inputBuffer  = getMTLBufferStorage(inputTensor);
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(outputTensor);
  MPSStream* mpsStream = getCurrentMPSStream();
  id<MTLDevice> device = MPSDevice::getInstance()->device();

  dispatch_sync(mpsStream->queue(), ^(){
    @autoreleasepool {
      NSError* error = nil;
      constexpr uint32_t nOffsets = 3;
      const int64_t num_indices = index_size.size();
      const uint32_t numThreads = iter.numel();
      const uint32_t nDim = iter.ndim();
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

      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
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

      MTLFunctionConstantValues* constantValues = [[MTLFunctionConstantValues new] autorelease];
      [constantValues setConstantValue: &num_indices type:MTLDataTypeUInt atIndex:0];

      std::string indexFunction = getIndexFunctionName(inputTensor.scalar_type(), index_select, accumulate);
      id<MTLFunction> indexKernelFunction = MPSDevice::getInstance()->metalIndexingFunction(indexFunction, constantValues);
      id<MTLArgumentEncoder> argumentEncoder = [[indexKernelFunction newArgumentEncoderWithBufferIndex:0] autorelease];
      NSUInteger argumentBufferLength = argumentEncoder.encodedLength;
      id<MTLBuffer> indexAB = [[device newBufferWithLength:argumentBufferLength options:0] autorelease];
      [argumentEncoder setArgumentBuffer:indexAB offset:0];

      for (uint32_t idx = 0; idx < num_indices; idx++) {
        const Tensor& indexTensor = iter.tensor(idx+2);
        [argumentEncoder setBuffer: getMTLBufferStorage(indexTensor)
                            offset: indexTensor.storage_offset() * indexTensor.element_size()
                           atIndex: idx];
        TORCH_CHECK(indexTensor.scalar_type() == ScalarType::Long, "index(): Expected dtype int64 for Index");
      }

      // FIXME: PSO needs to be cached
      id<MTLComputePipelineState> indexSelectPSO = [[device newComputePipelineStateWithFunction: indexKernelFunction
                                                                                          error: &error] autorelease];
      TORCH_CHECK(indexSelectPSO, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

      for (uint32_t idx = 0; idx < num_indices; idx++) {
        const Tensor& indexTensor = iter.tensor(idx+2);
        [computeEncoder useResource:getMTLBufferStorage(indexTensor) usage:MTLResourceUsageRead];
      }

      [computeEncoder setComputePipelineState:indexSelectPSO];
      [computeEncoder setBuffer:indexAB offset:0 atIndex:0];
      [computeEncoder setBytes:index_size.data() length:sizeof(index_size[0]) * index_size.size() atIndex:1];
      [computeEncoder setBytes:index_stride.data() length:sizeof(index_stride[0]) * index_stride.size() atIndex:2];
      [computeEncoder setBuffer:kernelDataOffsets offset:0 atIndex:3];
      [computeEncoder setBuffer:inputBuffer offset:inputTensor.storage_offset() * inputTensor.element_size() atIndex:4];
      [computeEncoder setBuffer:outputBuffer offset:outputTensor.storage_offset() * outputTensor.element_size() atIndex:5];

      NSUInteger tgSize = indexSelectPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > numThreads)
          tgSize = numThreads;

      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreads: gridSize
                threadsPerThreadgroup: threadGroupSize];

      [computeEncoder endEncoding];
      mpsStream->commit(true);
    }
  });

  return true;
}

static void validateInputData(const TensorIteratorBase& iter, IntArrayRef index_size, IntArrayRef index_stride, const std::string& op, bool accumulate) {
  using namespace mps;

  int64_t num_indices = index_size.size();
  TORCH_CHECK(num_indices <= 16, "Current limit allows up to 16 indices to be used in MPS indexing kernels");

  AT_ASSERT(num_indices == index_stride.size());
  AT_ASSERT(num_indices == iter.ntensors() - 2);
  const Tensor& inputTensor = iter.tensor(1);

  if (accumulate) {
    // No atomic support for the rest of dtypes
    TORCH_CHECK(inputTensor.scalar_type() == ScalarType::Float ||
                inputTensor.scalar_type() == ScalarType::Int   ||
                inputTensor.scalar_type() == ScalarType::Bool);
  } else {
    TORCH_CHECK(c10::isIntegralType(inputTensor.scalar_type(), /*includesBool=*/true) ||
                inputTensor.scalar_type() == ScalarType::Float ||
                inputTensor.scalar_type() == ScalarType::Half,
                getMPSTypeString(inputTensor.scalar_type()) + std::string(" not supported for index.Tensor_out"));
  }
}

void index_kernel_mps(TensorIteratorBase& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  using namespace mps;
  @autoreleasepool {
    validateInputData(iter, index_size, index_stride, "index.Tensor_out", /*accumulate=*/false);
    dispatchIndexKernel(iter, index_size, index_stride, /*index_select=*/true, /*accumulate=*/false);
  }
}

void index_put_kernel_mps(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
  using namespace mps;
  @autoreleasepool {
    validateInputData(iter, index_size, index_stride, "index_put_impl", accumulate);
    dispatchIndexKernel(iter, index_size, index_stride, /*index_select=*/false, accumulate);
  }
}

static Tensor & masked_select_out_mps_impl(Tensor & result, const Tensor & self, const Tensor & mask) {
  NoNamesGuard guard;

  TORCH_CHECK(mask.scalar_type() == ScalarType::Byte || mask.scalar_type() == ScalarType::Bool,
              "masked_select: expected BoolTensor or ByteTensor for mask");
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "masked_select(): self and result must have the same scalar type");

  auto mask_temp = (mask.dim() == 0)
    ? c10::MaybeOwned<Tensor>::owned(mask.unsqueeze(0))
    : c10::MaybeOwned<Tensor>::borrowed(mask);
  auto self_temp = (self.dim() == 0)
    ? c10::MaybeOwned<Tensor>::owned(self.unsqueeze(0))
    : c10::MaybeOwned<Tensor>::borrowed(self);

  // Cannot reassign to mask_temp and self_temp here! if they are
  // owning and expand_outplace returns a borrow, the returned borrow
  // would dangle.
  auto mask_self_expanded = expand_outplace(*mask_temp, *self_temp);
  at::index_out(
      result, *std::get<1>(mask_self_expanded),
      c10::List<c10::optional<at::Tensor>>({*std::move(std::get<0>(mask_self_expanded))}));

  return result;
}

static
Tensor nonzero_fallback(const Tensor& self) {
  TORCH_WARN_ONCE("MPS: nonzero op is supported natively starting from macOS 13.0. ",
                  "Falling back on CPU. This may have performance implications.");

  return at::nonzero(self.to("cpu")).clone().to("mps");
}

Tensor& nonzero_out_mps(const Tensor& self, Tensor& out_){
  if (!is_macos_13_or_newer()) {
      Tensor out_fallback = nonzero_fallback(self);
      at::native::resize_output(out_, out_fallback.sizes());
      out_.copy_(out_fallback.to("mps"));
      return out_;
  }

  using namespace mps;
  const uint32_t maxDimensions = 16;

  TORCH_CHECK(self.numel() < std::numeric_limits<int>::max(), "nonzero is not supported for tensors with more than INT_MAX elements, \
  file a support request");
  TORCH_CHECK(out_.dtype() == at::kLong, "Expected object of scalar type ", at::kLong, " as out, but got ", out_.dtype());
  TORCH_CHECK(self.device() == out_.device(), "expected self and out to be on the same device, but got out on ",
  out_.device(), " and self on ", self.device());
  TORCH_CHECK(self.dim() <= maxDimensions, "nonzero is not supported for tensor with more than ", 16, " dimensions");
  TORCH_CHECK(out_.is_mps());

  MPSStream *stream = getCurrentMPSStream();
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
    MPSGraphTensor* scatterDataTensor_ = nil;
  };

  int64_t total_nonzero = at::count_nonzero(self).item<int64_t>();
  int64_t nDim = self.dim();
  at::native::resize_output(out_, {total_nonzero, nDim});
  if (out_.numel() ==  0) {
    return out_;
  }

  bool contiguous_output = (out_.is_contiguous() && !out_.is_view());
  Tensor out = out_;
  if (!contiguous_output) {
    out = at::native::empty_mps(
           out_.sizes(),
           out_.scalar_type(),
           c10::nullopt,
           kMPS,
           c10::nullopt,
           c10::nullopt);
  }

  int64_t _apparentInputShape = 1;
  for (auto dim : self.sizes()) {
    _apparentInputShape *= dim;
  }
  MPSShape *apparentOutputShape = @[@(total_nonzero * nDim)];
  MPSShape *apparentInputShape = @[@(_apparentInputShape)];

  // Pseudocode:
  //
  // inputTensor     = [1,  0,  0,  3]
  // inputNonZero    = [1,  0,  0,  1]
  // indices         = [1,  1,  1,  2]
  // maskedIndices   = [0, -1, -1,  1]
  // coordinates     = [0,  1,  2,  3]
  // scatterResult   = [0,  3]

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  @autoreleasepool {
    string key = "nonzero_out_mps" + getTensorsStringKey(self);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;
        @autoreleasepool {
          MPSDataType inputDataType = getMPSDataType(self.scalar_type());
          MPSShape* inputShape = getMPSShape(self);
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor *inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(self.scalar_type()), apparentInputShape);
          MPSGraphTensor *scatterDataTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSScalarType(out.scalar_type()));
          MPSGraphTensor *zeroTensor = [mpsGraph constantWithScalar:0.0 dataType:inputDataType];
          MPSGraphTensor *oneTensor = [mpsGraph constantWithScalar:1.0 dataType:MPSDataTypeInt32];
          MPSGraphTensor *minusMaxDimTensor = [mpsGraph constantWithScalar:-maxDimensions dataType:MPSDataTypeInt32];
          MPSGraphTensor *inputNotEqualToZeroTensor = [mpsGraph notEqualWithPrimaryTensor:inputTensor
                                                                          secondaryTensor:zeroTensor
                                                                                     name:nil];
          MPSGraphTensor *maskTensor = [mpsGraph castTensor:inputNotEqualToZeroTensor
                                                     toType:MPSDataTypeInt32
                                                       name:@"castToInt32"];
          MPSGraphTensor *indicesTensor = [mpsGraph cumulativeSumWithTensor:maskTensor
                                                                       axis:0
                                                                       name:nil];
          MPSGraphTensor *indicesMinusOneTensor = [mpsGraph subtractionWithPrimaryTensor:indicesTensor
                                                                        secondaryTensor:oneTensor
                                                                                   name:nil];
          MPSGraphTensor *maskedIndicesTensor = [mpsGraph selectWithPredicateTensor:inputNotEqualToZeroTensor
                                                                truePredicateTensor:indicesMinusOneTensor
                                                               falsePredicateTensor:minusMaxDimTensor
                                                                               name:nil];
          MPSGraphTensor *coordinatesTensor = [mpsGraph reshapeTensor:[mpsGraph coordinateAlongAxis:0 withShape:inputShape name:nil]
                                                            withShape:@[@-1]
                                                                name:nil];
          if (nDim > 1) {
            NSMutableArray<MPSGraphTensor*> *maskedIndicesTensorArray = [NSMutableArray arrayWithCapacity:nDim];
            NSMutableArray<MPSGraphTensor*> *coordinatesTensorArray = [NSMutableArray arrayWithCapacity:nDim];

            MPSGraphTensor *constantRankTensor = [mpsGraph constantWithScalar:nDim
                                                                     dataType:MPSDataTypeInt32];
            maskedIndicesTensorArray[0] = [mpsGraph multiplicationWithPrimaryTensor:maskedIndicesTensor
                                                                    secondaryTensor:constantRankTensor
                                                                               name:nil];
            coordinatesTensorArray[0] = coordinatesTensor;
            for (int i = 1; i < nDim; i++){
              maskedIndicesTensorArray[i] = [mpsGraph additionWithPrimaryTensor:maskedIndicesTensorArray[i - 1]
                                                                secondaryTensor:oneTensor
                                                                           name:nil];
              coordinatesTensorArray[i] = [mpsGraph reshapeTensor:[mpsGraph coordinateAlongAxis:i withShape:inputShape name:nil]
                                                        withShape:@[@-1]
                                                             name:nil];
            }
            maskedIndicesTensor = [mpsGraph concatTensors:maskedIndicesTensorArray dimension:0 interleave:YES name:nil];
            coordinatesTensor = [mpsGraph concatTensors:coordinatesTensorArray dimension:0 interleave:YES name:nil];
          }

          MPSGraphTensor *outputTensor = [mpsGraph scatterWithDataTensor:scatterDataTensor
                                                           updatesTensor:coordinatesTensor
                                                           indicesTensor:maskedIndicesTensor
                                                                    axis:0
                                                                    mode:MPSGraphScatterModeSet
                                                                    name:nil];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->scatterDataTensor_ = scatterDataTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, apparentInputShape);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, contiguous_output ? out_ : out, apparentOutputShape);
    Placeholder scatterPlaceholder = Placeholder(cachedGraph->scatterDataTensor_, contiguous_output ? out_ : out, apparentOutputShape);

    // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      scatterPlaceholder.getMPSGraphTensor() : scatterPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    if (!contiguous_output) {
      out_.copy_(out);
    }
  }

  return out_;
}

Tensor nonzero_mps(const Tensor& self){
  if (!is_macos_13_or_newer()) {
    return nonzero_fallback(self);
  }

  Tensor out = at::empty({0}, self.options().dtype(kLong));
  return nonzero_out_mps(self, out);
}

Tensor masked_select_mps(const Tensor & self, const Tensor & mask) {
  namedinference::compute_broadcast_outnames(self, mask);
  Tensor result = at::empty({0}, self.options());
  return masked_select_out_mps_impl(result, self, mask);
}

Tensor & masked_select_out_mps(const Tensor & self, const Tensor & mask, Tensor & result) {
  namedinference::compute_broadcast_outnames(self, mask);
  return masked_select_out_mps_impl(result, self, mask);
}

Tensor flip_mps(const Tensor& self, IntArrayRef dims) {
  using namespace mps;

  Tensor result = at::native::empty_mps(
                    self.sizes(),
                    self.scalar_type(),
                    c10::nullopt,
                    kMPS,
                    c10::nullopt,
                    c10::nullopt);

  auto total_dims = self.dim();
  // It wraps the dims and checks that there are no repeated dims
  auto flip_dims_b = at::dim_list_to_bitset(dims, total_dims);
  NSMutableArray<NSNumber*> * ns_dims = [[NSMutableArray<NSNumber*> new] autorelease];

  for (const auto i : c10::irange(total_dims)) {
    if(flip_dims_b[i] && self.size(i) > 1 && self.stride(i) != 0) {
      [ns_dims addObject:[NSNumber numberWithInt:i]];
    }
  }

  // Nothing to do, we return fast
  if (dims.size() == 0 || self.numel() <=1) {
    result.copy_(self);
    return result;
  }

  MPSStream* stream = getCurrentMPSStream();

  using CachedGraph = mps::MPSUnaryCachedGraph;

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    NSString* ns_dims_key = [[ns_dims valueForKey:@"description"] componentsJoinedByString:@","];
    // A key is used to identify the MPSGraph which was created once, and can be reused if the parameters, data types etc match the earlier created MPSGraph
    string key = "flip_mps:" + getTensorsStringKey({self}) + ":" + string([ns_dims_key UTF8String]);
    auto cachedGraph = cache_->LookUpAs<CachedGraph>(key);
    if(!cachedGraph) {
      cachedGraph = cache_->CreateCachedGraphAs<CachedGraph>(key, ^ MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* outputTensor = [mpsGraph reverseTensor:inputTensor
                                                            axes:ns_dims
                                                            name:nil];
          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
    }

    // Create placeholders which use the keys of the CachedGraph to create inputs and outputs of the operation
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);


    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    // Run the graph
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return result;

}

TORCH_IMPL_FUNC(index_add_mps_out)(
  const Tensor& self,
  int64_t dim,
  const Tensor& index,
  const Tensor& source,
  const Scalar& alpha,
  const Tensor& result) {

  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();
  dim = maybe_wrap_dim(dim, self.dim());
  if (index.numel() == 0) {
    return;
  }

  TORCH_CHECK(source.scalar_type() != ScalarType::Long, "index_add(): Expected non int64 dtype for source.");
  auto casted_type = isFloatingType(source.scalar_type()) ? ScalarType::Float : ScalarType::Int;

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* indexTensor_ = nil;
    MPSGraphTensor* sourceTensor_ = nil;
    MPSGraphTensor* alphaTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {

    string key = "index_add_mps_out" + getTensorsStringKey({self, index, source}) + ":" + std::to_string(dim);
    CachedGraph* cachedGraph = cache_->LookUpAs<CachedGraph>(key);

    if(!cachedGraph) {
      cachedGraph = cache_->CreateCachedGraphAs<CachedGraph>(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* indexTensor = mpsGraphRankedPlaceHolder(mpsGraph, index);
          MPSGraphTensor* sourceTensor = mpsGraphRankedPlaceHolder(mpsGraph, source);
          MPSGraphTensor* alphaTensor = mpsGraphScalarPlaceHolder(mpsGraph, getMPSScalarType(casted_type));
          MPSGraphTensor* castedInputTensor = inputTensor;
          MPSGraphTensor* castedSourceTensor = sourceTensor;
          if (source.scalar_type() != casted_type) {
              castedInputTensor = castMPSTensor(mpsGraph, castedInputTensor, casted_type);
              castedSourceTensor = castMPSTensor(mpsGraph, castedSourceTensor, casted_type);
          }
          MPSGraphTensor* alphaSourceSlice = [mpsGraph multiplicationWithPrimaryTensor:castedSourceTensor
                                                                       secondaryTensor:alphaTensor
                                                                                  name:nil];

          MPSGraphTensor* outputTensor = [mpsGraph scatterWithDataTensor:castedInputTensor
                                                            updatesTensor:alphaSourceSlice
                                                            indicesTensor:indexTensor
                                                                     axis:dim
                                                                     mode:MPSGraphScatterModeAdd
                                                                     name:nil];
          if (source.scalar_type() != casted_type) {
              outputTensor = castMPSTensor(mpsGraph, outputTensor, source.scalar_type());
          }
          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->indexTensor_ = indexTensor;
          newCachedGraph->sourceTensor_ = sourceTensor;
          newCachedGraph->alphaTensor_ = alphaTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder indexPlaceholder = Placeholder(cachedGraph->indexTensor_, index);
    Placeholder sourcePlaceholder = Placeholder(cachedGraph->sourceTensor_, source);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);
    MPSScalar alpha_scalar = getMPSScalar(alpha, casted_type);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      indexPlaceholder.getMPSGraphTensor() : indexPlaceholder.getMPSGraphTensorData(),
      sourcePlaceholder.getMPSGraphTensor() : sourcePlaceholder.getMPSGraphTensorData(),
      cachedGraph->alphaTensor_ : getMPSGraphTensorFromScalar(stream, alpha_scalar),
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}

Tensor index_select_mps(const Tensor & self,
                         int64_t dim,
                         const Tensor & index) {
  IntArrayRef input_shape = self.sizes();
  auto num_input_dims = input_shape.size();

  auto num_indices = index.numel();
  TORCH_CHECK_INDEX(index.dim() <= 1, "index_select(): Index is supposed to be a vector");

  dim = maybe_wrap_dim(dim, self.dim());
  std::vector<int64_t> shape_data(num_input_dims);

  // Calculate new shape
  for(auto i : c10::irange(num_input_dims)) {
    if (i == dim) {
      shape_data[i] = num_indices;
    } else {
      shape_data[i] = input_shape[i];
    }
  }

  IntArrayRef output_shape = IntArrayRef(shape_data.data(), num_input_dims);

  Tensor result = at::native::empty_mps(
                      output_shape,
                      self.scalar_type(),
                      c10::nullopt,
                      kMPS,
                      c10::nullopt,
                      c10::nullopt);

  index_select_out_mps(self, dim, index, result);
  return result;
}

Tensor& index_select_out_mps(const Tensor & self,
                             int64_t dim,
                             const Tensor & index,
                             Tensor & output) {

  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();
  dim = maybe_wrap_dim(dim, self.dim());
  // Checks
  TORCH_CHECK_INDEX(index.dim() <= 1, "index_select(): Index is supposed to be a vector");
  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int, "index_select(): Expected dtype int32 or int64 for index");
  TORCH_CHECK(self.scalar_type() == output.scalar_type(),
              "index_select(): self and output must have the same scalar type");
  TORCH_CHECK(dim == 0 || dim < self.dim(),
              "index_select(): Indexing dim ", dim, " is out of bounds of tensor");

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* indexTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  auto inputType = getMPSDataType(self.scalar_type());
  auto outputType = getMPSDataType(output.scalar_type());
  if (inputType == MPSDataTypeUInt8 || inputType == MPSDataTypeBool) {
      inputType = MPSDataTypeInt8;
  }
  if (outputType == MPSDataTypeUInt8 || outputType == MPSDataTypeBool) {
      outputType = MPSDataTypeInt8;
  }
  @autoreleasepool {

    string key = "index_select_out_mps" + getTensorsStringKey({self, index}) + ":" + std::to_string(dim);
    CachedGraph* cachedGraph = cache_->LookUpAs<CachedGraph>(key);

    if(!cachedGraph) {
      cachedGraph = cache_->CreateCachedGraphAs<CachedGraph>(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, inputType, getMPSShape(self));
          MPSGraphTensor* indexTensor = mpsGraphRankedPlaceHolder(mpsGraph, index);

          MPSGraphTensor* outputTensor = [mpsGraph gatherWithUpdatesTensor:inputTensor
                                                             indicesTensor:indexTensor
                                                                      axis:dim
                                                           batchDimensions:0
                                                                      name:nil];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->indexTensor_ = indexTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self,
                                  /*mpsShape=*/nullptr, /*gatherTensorData=*/true, /*dataType=*/inputType);
    Placeholder indexPlaceholder = Placeholder(cachedGraph->indexTensor_, index);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output,
                                  /*mpsShape=*/nullptr, /*gatherTensorData=*/false, /*dataType=*/outputType);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      indexPlaceholder.getMPSGraphTensor() : indexPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return output;

}

Tensor & masked_fill__mps(Tensor& self, const Tensor & mask, const Scalar& value) {
  using namespace mps;
  TORCH_CHECK(self.device() == mask.device(), "expected self and mask to be on the same device, but got mask on ",
    mask.device(), " and self on ", self.device());
  TORCH_CHECK(mask.scalar_type() == kByte || mask.scalar_type() == kBool,
    "expected mask dtype to be Bool but got ", mask.scalar_type());
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *maskTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  MPSStream* stream = getCurrentMPSStream();
  @autoreleasepool {
    string key = "masked_fill" + getTensorsStringKey({self, mask}) + ":" + std::to_string(value.toDouble());
    CachedGraph* cachedGraph = cache_->LookUpAs<CachedGraph>(key);
    if(!cachedGraph) {
      cachedGraph = cache_->CreateCachedGraphAs<CachedGraph>(key, ^ MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* maskTensor = mpsGraphRankedPlaceHolder(mpsGraph, mask);
          MPSDataType valueType = getMPSScalarType(value.type());

          // constantWithScalar doesn't like Bool constants getting created so
          // mapping them to int8
          if (valueType == MPSDataTypeBool) {
            valueType = MPSDataTypeInt8;
          }
          MPSGraphTensor* valueTensor =  [mpsGraph constantWithScalar:value.to<double>()
                                                            dataType:valueType];
          valueTensor = [mpsGraph castTensor:valueTensor
                                          toType:getMPSDataType(self.scalar_type())
                                           name : @"castTensorEq"];

          MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:maskTensor
                                                        truePredicateTensor:valueTensor
                                                        falsePredicateTensor:inputTensor
                                                             name:nil];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->maskTensor_ = maskTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
    }

    Placeholder selfPlaceholder   = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder maskPlaceholder   = Placeholder(cachedGraph->maskTensor_, mask);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, self);

    // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      maskPlaceholder.getMPSGraphTensor() : maskPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);

  }
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

Tensor embedding_dense_backward_mps(
    const Tensor & grad_, const Tensor & indices, int64_t num_weights,
    int64_t padding_idx, bool scale_grad_by_freq)
{
    // TODO: implement padding_idx & scale_grad_by_freq.
    namespace native_mps = at::native::mps;
    struct CachedGraph : public native_mps::MPSCachedGraph
    {
      CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
      MPSGraphTensor *incomingGradTensor_ = nil;
      MPSGraphTensor *indicesTensor_ = nil;
      MPSGraphTensor *outgoingGradTensor_ = nil;
    };

    native_mps::MPSGraphCache* cache_ = native_mps::MPSGraphCache::getInstance();

    IntArrayRef incoming_gradient_shape = grad_.sizes();
    int64_t num_incoming_gradient_dims = incoming_gradient_shape.size();

    IntArrayRef indices_shape = indices.sizes();
    int64_t num_indices_dims = indices_shape.size();

    int64_t D = incoming_gradient_shape[num_incoming_gradient_dims - 1];
    c10::SmallVector<int64_t, 2> outgoing_gradient_shape{num_weights, D};
    Tensor outgoing_gradient = at::native::empty_mps(
                                IntArrayRef(outgoing_gradient_shape),
                                grad_.scalar_type(),
                                c10::nullopt,
                                kMPS,
                                c10::nullopt,
                                c10::nullopt);

    if (outgoing_gradient.numel() == 0) {
      return outgoing_gradient;
    }

    auto stream = at::mps::getCurrentMPSStream();

    @autoreleasepool {
        string key = "edb_mps:" + native_mps::getMPSTypeString(grad_.scalar_type()) + ":indices" + std::to_string(num_indices_dims) + ":num_weights" + std::to_string(num_weights) + ":padding_idx" + std::to_string(padding_idx) + ":scaled" + std::to_string(scale_grad_by_freq);
      CachedGraph* cachedGraph = cache_->LookUpAs<CachedGraph>(key);
      // Initialize once if configuration not found in cache
      if(!cachedGraph) {
        cachedGraph = cache_->CreateCachedGraphAs<CachedGraph>(key, ^ native_mps::MPSCachedGraph * () {

          CachedGraph *newCachedGraph = nil;

          @autoreleasepool {
            MPSGraph* mpsGraph = native_mps::make_mps_graph();
            newCachedGraph = new CachedGraph(mpsGraph);

            MPSGraphTensor* incomingGradTensor = native_mps::mpsGraphUnrankedPlaceHolder(mpsGraph, native_mps::getMPSDataType(grad_.scalar_type()));

            MPSGraphTensor* indicesTensor = native_mps::mpsGraphUnrankedPlaceHolder(mpsGraph, native_mps::getMPSDataType(indices.scalar_type()));

            MPSGraphTensor* reshapedIndicesTensor = indicesTensor;

            if (num_indices_dims != 0) {
              reshapedIndicesTensor = [mpsGraph  expandDimsOfTensor: indicesTensor
                                                               axes: @[@-1]
                                                               name: nil];
            }

            auto outgoingGradTensor = [mpsGraph scatterNDWithUpdatesTensor: incomingGradTensor
                                                             indicesTensor: reshapedIndicesTensor
                                                                     shape: native_mps::getMPSShape(IntArrayRef(outgoing_gradient_shape))
                                                           batchDimensions: 0
                                                                      mode: MPSGraphScatterModeAdd
                                                                      name: @"edb"];

            newCachedGraph->incomingGradTensor_ = incomingGradTensor;
            newCachedGraph->indicesTensor_ = indicesTensor;
            newCachedGraph->outgoingGradTensor_ = outgoingGradTensor;

          }
          return newCachedGraph;
        });
      }
      auto incomingGradPlaceholder = native_mps::Placeholder(cachedGraph->incomingGradTensor_, grad_);
      auto indicesPlaceholder = native_mps::Placeholder(cachedGraph->indicesTensor_, indices);
      auto outgoingGradPlaceholder = native_mps::Placeholder(cachedGraph->outgoingGradTensor_, outgoing_gradient);

      NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
          incomingGradPlaceholder.getMPSGraphTensor() : incomingGradPlaceholder.getMPSGraphTensorData(),
          indicesPlaceholder.getMPSGraphTensor() : indicesPlaceholder.getMPSGraphTensorData()
      };

      NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
          outgoingGradPlaceholder.getMPSGraphTensor() : outgoingGradPlaceholder.getMPSGraphTensorData()
      };
      native_mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    }
    return outgoing_gradient;
}

Tensor & masked_fill__mps(Tensor& self, const Tensor & mask, const Tensor & value) {
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", value.dim(), " dimension(s).");
  return masked_fill__mps(self, mask, value.item());
}

REGISTER_DISPATCH(index_stub, &index_kernel_mps);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel_mps);
} // namespace at::native
