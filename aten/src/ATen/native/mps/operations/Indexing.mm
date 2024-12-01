//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>

#include <ATen/AccumulateType.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/ceil_div.h>
#include <ATen/core/TensorBody.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/operations/Indexing.h>
#include <c10/core/QScheme.h>
#include <c10/util/SmallVector.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/count_nonzero.h>
#include <ATen/ops/count_nonzero_native.h>
#include <ATen/ops/embedding_dense_backward_native.h>
#include <ATen/ops/flip_native.h>
#include <ATen/ops/index.h>
#include <ATen/ops/index_add_native.h>
#include <ATen/ops/index_fill_native.h>
#include <ATen/ops/index_put.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/masked_fill_native.h>
#include <ATen/ops/masked_scatter_native.h>
#include <ATen/ops/masked_select_native.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/nonzero_native.h>
#include <ATen/ops/view_as_real.h>
#endif

constexpr auto nonZeroMaxSize = 1UL << 24;

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Indexing_metallib.h>
#endif

id<MTLBuffer> generateKernelDataOffsets(id<MTLComputeCommandEncoder> commandEncoder,
                                        const TensorIteratorBase& iter,
                                        bool use_64bit_index) {
  constexpr uint32_t nOffsets = 3;
  uint32_t numThreads = iter.numel();
  const uint32_t nDim = iter.ndim();
  const IntArrayRef& iterShape = iter.shape();
  std::vector<uint32_t> iterShapeData(iterShape.size());
  std::vector<std::array<uint32_t, nOffsets>> strides(nDim);
  TORCH_INTERNAL_ASSERT(iter.ntensors() >= nOffsets);
  TORCH_CHECK(use_64bit_index || iter.can_use_32bit_indexing(), "Can't be indexed using 32-bit iterator");

  for (const auto i : c10::irange(iterShape.size())) {
    iterShapeData[i] = static_cast<uint32_t>(iterShape[i]);
  }

  for (const auto i : c10::irange(nDim)) {
    for (const auto offset : c10::irange(nOffsets)) {
      strides[i][offset] = static_cast<uint32_t>(iter.strides(offset)[i]);
    }
  }

  auto kernelDataOffsetsPSO =
      lib.getPipelineStateForFunc(use_64bit_index ? "kernel_index_offsets_64" : "kernel_index_offsets_32");
  const auto elementSize = use_64bit_index ? sizeof(simd_ulong3) : sizeof(simd_uint3);
  id<MTLBuffer> kernelDataOffsets = (id<MTLBuffer>)getIMPSAllocator()->allocate(numThreads * elementSize).get();

  [commandEncoder setComputePipelineState:kernelDataOffsetsPSO];
  [commandEncoder setBytes:strides.data() length:sizeof(uint32_t) * nDim * nOffsets atIndex:0];
  [commandEncoder setBuffer:kernelDataOffsets offset:0 atIndex:1];
  [commandEncoder setBytes:iterShapeData.data() length:sizeof(uint32_t) * iterShape.size() atIndex:2];
  [commandEncoder setBytes:&nDim length:sizeof(uint32_t) atIndex:3];

  mtl_dispatch1DJob(commandEncoder, kernelDataOffsetsPSO, numThreads);

  return kernelDataOffsets;
}

static std::string getBitSizeString(ScalarType scalar_type) {
  size_t scalarBitSize = c10::elementSize(scalar_type) * 8;
  TORCH_CHECK(scalarBitSize <= 64, "Unsupported data type: ", getMPSTypeString(scalar_type));
  return std::to_string(scalarBitSize) + "bit";
}
static std::string getIndexFunctionName(ScalarType scalar_type,
                                        bool index_select,
                                        bool accumulate,
                                        bool serial,
                                        bool use_64bit_indexing) {
  std::string indexFunction = index_select     ? "index_select_"
      : (accumulate && (scalar_type != kBool)) ? "index_put_accumulate_"
                                               : (serial ? "index_put_serial_" : "index_put_");

  indexFunction += getBitSizeString(scalar_type);
  if (accumulate) {
    TORCH_CHECK(scalar_type == ScalarType::Float || scalar_type == ScalarType::Int,
                "Unsupported data type for accumulate case: ",
                getMPSTypeString(scalar_type));
    string dtypeString = (scalar_type == ScalarType::Float) ? "_float" : "_int";
    indexFunction += dtypeString;
  }
  indexFunction += use_64bit_indexing ? "_idx64" : "_idx32";
  return indexFunction;
}

static bool dispatchIndexKernel(TensorIteratorBase& iter,
                                IntArrayRef index_size,
                                IntArrayRef index_stride,
                                bool index_select,
                                bool accumulate) {
  using namespace mps;

  if (iter.numel() == 0) {
    return true;
  }
  const bool serial_index_put = at::globalContext().deterministicAlgorithms() && !accumulate && !index_select;

  const Tensor& inputTensor = iter.tensor(1);
  Tensor outputTensor = iter.tensor(0);
  MPSStream* mpsStream = getCurrentMPSStream();
  id<MTLDevice> device = MPSDevice::getInstance()->device();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      NSError* error = nil;
      const int64_t num_indices = index_size.size();
      const uint32_t numIters = serial_index_put ? iter.numel() : 1;
      uint32_t numThreads = iter.numel();

      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const bool use_64bit_indexing = !iter.can_use_32bit_indexing();
      auto kernelDataOffsets = generateKernelDataOffsets(computeEncoder, iter, use_64bit_indexing);

      auto indexFunction = getIndexFunctionName(
          inputTensor.scalar_type(), index_select, accumulate, serial_index_put, use_64bit_indexing);
      auto indexSelectPSO = lib.getPipelineStateForFunc(indexFunction);
      size_t argumentBufferLength = sizeof(uint64_t) * num_indices;
      auto indexAB = [[device newBufferWithLength:argumentBufferLength options:0] autorelease];
      uint64_t* indexABContents = (uint64_t*)(indexAB.contents);
      for (uint32_t idx = 0; idx < num_indices; idx++) {
        const Tensor& indexTensor = iter.tensor(idx + 2);
        indexABContents[idx] =
            getMTLBufferStorage(indexTensor).gpuAddress + (indexTensor.storage_offset() * indexTensor.element_size());
        TORCH_CHECK(indexTensor.scalar_type() == ScalarType::Long, "index(): Expected dtype int64 for Index");
        [computeEncoder useResource:getMTLBufferStorage(indexTensor) usage:MTLResourceUsageRead];
      }
      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(indexSelectPSO, indexFunction, {inputTensor});

      [computeEncoder setComputePipelineState:indexSelectPSO];
      mtl_setArgs(
          computeEncoder, indexAB, index_size, index_stride, kernelDataOffsets, inputTensor, outputTensor, num_indices);
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      if (serial_index_put) {
        mtl_setBytes(computeEncoder, numIters, 7);
        gridSize = MTLSizeMake(1, 1, 1);
        numThreads = 1;
      }

      NSUInteger tgSize = indexSelectPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > numThreads) {
        tgSize = numThreads;
      }

      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(indexSelectPSO);
    }
  });

  return true;
}

static void validateInputData(const TensorIteratorBase& iter,
                              IntArrayRef index_size,
                              IntArrayRef index_stride,
                              const std::string& op,
                              bool accumulate) {
  using namespace mps;

  const auto num_indices = index_size.size();
  TORCH_CHECK(num_indices <= 16, "Current limit allows up to 16 indices to be used in MPS indexing kernels");

  AT_ASSERT(num_indices == index_stride.size());
  AT_ASSERT(static_cast<int>(num_indices) == iter.ntensors() - 2);
  const Tensor& inputTensor = iter.tensor(1);
  const auto scalar_type = inputTensor.scalar_type();

  if (accumulate) {
    // No atomic support for the rest of dtypes
    TORCH_CHECK(scalar_type == ScalarType::Float || inputTensor.scalar_type() == ScalarType::Int ||
                scalar_type == ScalarType::Bool);
  } else {
    TORCH_CHECK(c10::isIntegralType(scalar_type, /*includesBool=*/true) || supportedFloatingType(scalar_type) ||
                    scalar_type == ScalarType::ComplexFloat || scalar_type == ScalarType::ComplexHalf,
                getMPSTypeString(inputTensor) + std::string(" not supported for index.Tensor_out"));
  }
}

static Tensor& masked_select_out_mps_impl(Tensor& result, const Tensor& self, const Tensor& mask) {
  NoNamesGuard guard;

  TORCH_CHECK(mask.scalar_type() == ScalarType::Bool, "masked_select: expected BoolTensor for mask");
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "masked_select(): self and result must have the same scalar type");

  auto mask_temp =
      (mask.dim() == 0) ? c10::MaybeOwned<Tensor>::owned(mask.unsqueeze(0)) : c10::MaybeOwned<Tensor>::borrowed(mask);
  auto self_temp =
      (self.dim() == 0) ? c10::MaybeOwned<Tensor>::owned(self.unsqueeze(0)) : c10::MaybeOwned<Tensor>::borrowed(self);

  // Cannot reassign to mask_temp and self_temp here! if they are
  // owning and expand_outplace returns a borrow, the returned borrow
  // would dangle.
  auto mask_self_expanded = expand_outplace(*mask_temp, *self_temp);
  at::index_out(result,
                *std::get<1>(mask_self_expanded),
                c10::List<std::optional<at::Tensor>>({*std::move(std::get<0>(mask_self_expanded))}));

  return result;
}

static void index_kernel_mps(TensorIteratorBase& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  @autoreleasepool {
    validateInputData(iter, index_size, index_stride, "index.Tensor_out", /*accumulate=*/false);
    dispatchIndexKernel(iter, index_size, index_stride, /*index_select=*/true, /*accumulate=*/false);
  }
}

static void index_put_kernel_mps(TensorIterator& iter,
                                 IntArrayRef index_size,
                                 IntArrayRef index_stride,
                                 bool accumulate) {
  @autoreleasepool {
    validateInputData(iter, index_size, index_stride, "index_put_impl", accumulate);
    dispatchIndexKernel(iter, index_size, index_stride, /*index_select=*/false, accumulate);
  }
}
} // namespace mps

static Tensor nonzero_fallback(const Tensor& self) {
  return at::nonzero(self.to("cpu")).to("mps");
}

static Tensor& nonzero_out_native_mps(const Tensor& self, Tensor& out_) {
  using namespace mps;

  int64_t nDim = self.dim();
  MPSStream* stream = getCurrentMPSStream();
  using CachedGraph = MPSUnaryCachedGraph;

  dispatch_sync(stream->queue(), ^() {
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
  });
  int64_t total_nonzero = at::count_nonzero(self).item<int64_t>();
  at::native::resize_output(out_, {total_nonzero, nDim});
  if (out_.numel() == 0) {
    return out_;
  }

  bool contiguous_output = !needsGather(out_);
  Tensor out = out_;
  if (!contiguous_output) {
    out = at::empty_like(out_, MemoryFormat::Contiguous);
  }

  @autoreleasepool {
    string key = "nonzero_out_native_mps" + getTensorsStringKey(self);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* outputTensor = [mpsGraph nonZeroIndicesOfTensor:inputTensor name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (!contiguous_output) {
    out_.copy_(out);
  }

  return out_;
}

Tensor& nonzero_out_mps(const Tensor& self, Tensor& out_) {
  if (!is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS)) {
    TORCH_WARN_ONCE("MPS: nonzero op is supported natively starting from macOS 14.0. ",
                    "Falling back on CPU. This may have performance implications.");
    Tensor out_fallback = nonzero_fallback(self);
    at::native::resize_output(out_, out_fallback.sizes());
    out_.copy_(out_fallback);
    return out_;
  } else if (self.is_complex()) {
    TORCH_WARN_ONCE("MPS: nonzero op is not supported for complex datatypes. ",
                    "Falling back on CPU. This may have performance implications.");
    Tensor out_fallback = nonzero_fallback(self);
    at::native::resize_output(out_, out_fallback.sizes());
    out_.copy_(out_fallback);
    return out_;
  }

  int64_t nDim = self.dim();
  if (self.numel() == 0) {
    at::native::resize_output(out_, {0, nDim});
    return out_;
  }

  using namespace mps;
  const uint32_t maxDimensions = 16;

  TORCH_CHECK(self.numel() < std::numeric_limits<int>::max(),
              "nonzero is not supported for tensors with more than INT_MAX elements, \
  See https://github.com/pytorch/pytorch/issues/51871");
  TORCH_CHECK(
      out_.dtype() == at::kLong, "Expected object of scalar type ", at::kLong, " as out, but got ", out_.dtype());
  TORCH_CHECK(self.device() == out_.device(),
              "expected self and out to be on the same device, but got out on ",
              out_.device(),
              " and self on ",
              self.device());
  TORCH_CHECK(self.dim() <= maxDimensions, "nonzero is not supported for tensor with more than ", 16, " dimensions");
  TORCH_CHECK(out_.is_mps());

  if (!is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS) &&
      (self.numel() >= nonZeroMaxSize || self.is_complex())) {
    TORCH_WARN_ONCE("MPS: nonzero op is not natively supported for the provided input on MacOS14",
                    "Falling back on CPU. This may have performance implications.",
                    "See github.com/pytorch/pytorch/issues/122916 for further info");
    Tensor out_fallback = nonzero_fallback(self);
    at::native::resize_output(out_, out_fallback.sizes());
    out_.copy_(out_fallback);
    return out_;
  }

  MPSStream* stream = getCurrentMPSStream();
  using CachedGraph = MPSUnaryCachedGraph;

  dispatch_sync(stream->queue(), ^() {
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
  });
  int64_t total_nonzero = at::count_nonzero(self).item<int64_t>();
  at::native::resize_output(out_, {total_nonzero, nDim});
  if (out_.numel() == 0) {
    return out_;
  }

  bool contiguous_output = !needsGather(out_);
  Tensor out = out_;
  if (!contiguous_output) {
    out = at::empty_like(out_, MemoryFormat::Contiguous);
  }

  @autoreleasepool {
    string key = "nonzero_out_native_mps" + getTensorsStringKey(self);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* outputTensor = [mpsGraph nonZeroIndicesOfTensor:inputTensor name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (!contiguous_output) {
    out_.copy_(out);
  }

  return out_;
}

Tensor nonzero_mps(const Tensor& self) {
  if (!is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS)) {
    TORCH_WARN_ONCE("MPS: nonzero op is supported natively starting from macOS 14.0. ",
                    "Falling back on CPU. This may have performance implications.");
    return nonzero_fallback(self);
  } else if (self.is_complex()) {
    TORCH_WARN_ONCE("MPS: nonzero op is not supported for complex datatypes ",
                    "Falling back on CPU. This may have performance implications.");
    return nonzero_fallback(self);
  }

  Tensor out = at::empty({0}, self.options().dtype(kLong));
  return nonzero_out_mps(self, out);
}

Tensor masked_select_mps(const Tensor& self, const Tensor& mask) {
  namedinference::compute_broadcast_outnames(self, mask);
  Tensor result = at::empty({0}, self.options());
  return mps::masked_select_out_mps_impl(result, self, mask);
}

Tensor& masked_select_out_mps(const Tensor& self, const Tensor& mask, Tensor& result) {
  namedinference::compute_broadcast_outnames(self, mask);
  return mps::masked_select_out_mps_impl(result, self, mask);
}

Tensor flip_mps(const Tensor& self, IntArrayRef dims) {
  using namespace mps;

  Tensor result = at::empty(self.sizes(), self.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);

  auto total_dims = self.dim();
  // It wraps the dims and checks that there are no repeated dims
  auto flip_dims_b = at::dim_list_to_bitset(dims, total_dims);
  NSMutableArray<NSNumber*>* ns_dims = [[NSMutableArray<NSNumber*> new] autorelease];

  for (const auto i : c10::irange(total_dims)) {
    if (flip_dims_b[i] && self.size(i) > 1 && self.stride(i) != 0) {
      [ns_dims addObject:[NSNumber numberWithInt:i]];
    }
  }

  // Nothing to do, we return fast
  if (self.numel() <= 1 || ns_dims.count == 0) {
    result.copy_(self);
    return result;
  }

  MPSStream* stream = getCurrentMPSStream();

  using CachedGraph = mps::MPSUnaryCachedGraph;

  MPSDataType inputDataType = getMPSScalarType(self.scalar_type());
  MPSDataType outputDataType = getMPSScalarType(self.scalar_type());
  @autoreleasepool {
    NSString* ns_dims_key = [[ns_dims valueForKey:@"description"] componentsJoinedByString:@","];
    // A key is used to identify the MPSGraph which was created once, and can be reused if the parameters, data types
    // etc match the earlier created MPSGraph
    string key = "flip_mps:" + getTensorsStringKey({self}) + ":" + string([ns_dims_key UTF8String]);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, inputDataType, getMPSShape(self));
      MPSGraphTensor* outputTensor = [mpsGraph reverseTensor:inputTensor axes:ns_dims name:nil];
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    // Create placeholders which use the keys of the CachedGraph to create inputs and outputs of the operation
    Placeholder inputPlaceholder =
        Placeholder(cachedGraph->inputTensor_, self, /*mpsShape*/ nil, /*gatherTensorData=*/true, inputDataType);
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor_, result, /*mpsShape*/ nil, /*gatherTensorData=*/false, outputDataType);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return result;
}

TORCH_IMPL_FUNC(index_add_mps_out)
(const Tensor& self,
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

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* indexTensor_ = nil;
    MPSGraphTensor* sourceTensor_ = nil;
    MPSGraphTensor* alphaTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  @autoreleasepool {
    string key = "index_add_mps_out" + getTensorsStringKey({self, index, source}) + ":" + std::to_string(dim);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
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
    });

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
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

Tensor index_select_mps(const Tensor& self, int64_t dim, const Tensor& index) {
  IntArrayRef input_shape = self.sizes();
  auto num_input_dims = input_shape.size();

  auto num_indices = index.numel();
  TORCH_CHECK_INDEX(index.dim() <= 1, "index_select(): Index is supposed to be a vector");

  dim = maybe_wrap_dim(dim, self.dim());
  std::vector<int64_t> shape_data(num_input_dims);

  // Calculate new shape
  for (const auto i : c10::irange(num_input_dims)) {
    if (i == static_cast<decltype(i)>(dim)) {
      shape_data[i] = num_indices;
    } else {
      shape_data[i] = input_shape[i];
    }
  }

  IntArrayRef output_shape = IntArrayRef(shape_data.data(), num_input_dims);

  Tensor result = at::empty(output_shape, self.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);

  index_select_out_mps(self, dim, index, result);
  return result;
}

Tensor& index_select_out_mps(const Tensor& self, int64_t dim, const Tensor& index, Tensor& output) {
  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();
  auto num_indices = index.numel();
  dim = maybe_wrap_dim(dim, self.dim());

  // Checks
  TORCH_CHECK_INDEX(index.dim() <= 1, "index_select(): Index is supposed to be a vector");
  TORCH_CHECK(!(self.dim() == 0 && num_indices != 1),
              "index_select(): Index to scalar can have only 1 value, got ",
              num_indices,
              " value(s)");
  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int,
              "index_select(): Expected dtype int32 or int64 for index");
  TORCH_CHECK(self.scalar_type() == output.scalar_type(),
              "index_select(): self and output must have the same scalar type");
  TORCH_CHECK(dim == 0 || dim < self.dim(), "index_select(): Indexing dim ", dim, " is out of bounds of tensor");
  TORCH_CHECK(output.dim() == 0 || index.size(-1) == output.size(dim),
              "index_select(): index and output must have the same size at `dim`th dimension, but got ",
              index.size(-1),
              " and ",
              output.size(dim),
              ".");

  for (const auto i : irange(self.dim())) {
    if (i == dim)
      continue;
    TORCH_CHECK(self.size(i) == output.size(i),
                "index_select(): self and output must have the same dimensions except for `dim`th dimension, but got ",
                self.size(i),
                " and ",
                output.size(i),
                " at dimension ",
                i,
                ".");
  }

  // Empty index
  if (num_indices == 0 || self.numel() == 0) {
    return output;
  }

  // Scalar input
  if (self.dim() == 0 && self.numel() == 1) {
    output.copy_(self);
    return output;
  }

  // As of MacOS 14.4 gatherWithUpdatesTensor: still does not support complex
  // So back to old view_as_real trick
  if (self.is_complex()) {
    auto out_view = at::view_as_real(output);
    index_select_out_mps(at::view_as_real(self), dim, index, out_view);
    return output;
  }

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* indexTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  auto inputType = getMPSDataType(self);
  auto outputType = getMPSDataType(output);
  if (inputType == MPSDataTypeUInt8) {
    inputType = MPSDataTypeInt8;
  }
  if (outputType == MPSDataTypeUInt8) {
    outputType = MPSDataTypeInt8;
  }

  @autoreleasepool {
    string key = "index_select_out_mps" + getTensorsStringKey({self, index}) + ":" + std::to_string(dim);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
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
    });

    // MPS TODO: MPS Gather is failing with MPS strided API. Fallback to old gather.
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_,
                                              self,
                                              /*mpsShape=*/nullptr,
                                              /*gatherTensorData=*/true,
                                              /*dataType=*/inputType,
                                              /*useStridedAPI=*/false);
    Placeholder indexPlaceholder = Placeholder(cachedGraph->indexTensor_, index, nil, true, MPSDataTypeInvalid, false);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_,
                                                output,
                                                /*mpsShape=*/nullptr,
                                                /*gatherTensorData=*/false,
                                                /*dataType=*/outputType,
                                                /*useStridedAPI=*/false);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder, indexPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return output;
}

Tensor& masked_fill__mps(Tensor& self, const Tensor& mask, const Scalar& value) {
  using namespace mps;

  if (self.numel() == 0) {
    return self;
  }
  TORCH_CHECK(self.device() == mask.device(),
              "expected self and mask to be on the same device, but got mask on ",
              mask.device(),
              " and self on ",
              self.device());
  TORCH_CHECK(mask.scalar_type() == kBool, "expected mask dtype to be Bool but got ", mask.scalar_type());
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");

  bool needs_output_copy = false;

  Tensor output;
  if (needsGather(self)) {
    output = at::empty(self.sizes(), self.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
    needs_output_copy = true;
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* maskTensor_ = nil;
    MPSGraphTensor* valueTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSDataType inputDataType = getMPSScalarType(self.scalar_type());
  MPSDataType maskDataType = getMPSScalarType(b_mask->scalar_type());

  MPSStream* stream = getCurrentMPSStream();
  MPSScalar valueScalar = getMPSScalar(value, value.type());
  @autoreleasepool {
    string key = "masked_fill" + getTensorsStringKey({self, *b_mask}) + ":" + getMPSTypeString(value.type());
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, inputDataType, getMPSShape(self));
      MPSGraphTensor* maskTensor = mpsGraphRankedPlaceHolder(mpsGraph, maskDataType, getMPSShape(*b_mask));
      MPSGraphTensor* valueTensor = mpsGraphScalarPlaceHolder(mpsGraph, value);

      MPSDataType valueType = getMPSScalarType(value.type());
      MPSGraphTensor* castValueTensor = valueTensor;
      if (valueType != inputDataType) {
        castValueTensor = [mpsGraph castTensor:valueTensor toType:inputDataType name:@"castValueTensor"];
      }

      MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:maskTensor
                                                     truePredicateTensor:castValueTensor
                                                    falsePredicateTensor:inputTensor
                                                                    name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->maskTensor_ = maskTensor;
      newCachedGraph->valueTensor_ = valueTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder =
        Placeholder(cachedGraph->inputTensor_, self, /*mpsShape*/ nil, /*gatherTensorData=*/true, inputDataType);
    Placeholder maskPlaceholder =
        Placeholder(cachedGraph->maskTensor_, *b_mask, /*mpsShape*/ nil, /*gatherTensorData=*/true, maskDataType);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_,
                                                needs_output_copy ? output : self,
                                                /*mpsShape*/ nil,
                                                /*gatherTensorData=*/false,
                                                inputDataType);

    // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      maskPlaceholder.getMPSGraphTensor() : maskPlaceholder.getMPSGraphTensorData(),
      cachedGraph->valueTensor_ : getMPSGraphTensorFromScalar(stream, valueScalar)
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (needs_output_copy) {
    self.copy_(output);
  }

  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

Tensor embedding_dense_backward_mps(const Tensor& grad_,
                                    const Tensor& indices,
                                    int64_t num_weights,
                                    int64_t padding_idx,
                                    bool scale_grad_by_freq) {
  // TODO: implement padding_idx & scale_grad_by_freq.
  using namespace at::native::mps;
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* incomingGradTensor_ = nil;
    MPSGraphTensor* indicesTensor_ = nil;
    MPSGraphTensor* outgoingGradTensor_ = nil;
  };

  IntArrayRef incoming_gradient_shape = grad_.sizes();
  int64_t num_incoming_gradient_dims = incoming_gradient_shape.size();

  IntArrayRef indices_shape = indices.sizes();
  int64_t num_indices_dims = indices_shape.size();

  int64_t D = incoming_gradient_shape[num_incoming_gradient_dims - 1];
  c10::SmallVector<int64_t, 2> outgoing_gradient_shape{num_weights, D};
  Tensor outgoing_gradient = at::empty(
      IntArrayRef(outgoing_gradient_shape), grad_.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);

  if (outgoing_gradient.numel() == 0) {
    return outgoing_gradient;
  }

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    string key = "edb_mps:" + getTensorsStringKey({grad_, indices}) + ":num_weights" + std::to_string(num_weights) +
        ":padding_idx" + std::to_string(padding_idx) + ":scaled" + std::to_string(scale_grad_by_freq);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* incomingGradTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(grad_));

      MPSGraphTensor* indicesTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(indices));

      MPSGraphTensor* reshapedIndicesTensor = indicesTensor;

      MPSGraphTensor* castGradTensor = incomingGradTensor;
      MPSDataType dataType = mps::getMPSDataType(grad_);
      // issue 105486100, scatterNDWithUpdatesTensor produces wrong result for float16
      if (dataType == MPSDataTypeFloat16) {
        castGradTensor = [mpsGraph castTensor:incomingGradTensor toType:MPSDataTypeFloat32 name:@"castGradTensor"];
      }
      if (num_indices_dims != 0) {
        reshapedIndicesTensor = [mpsGraph expandDimsOfTensor:indicesTensor axes:@[ @-1 ] name:nil];
      }

      auto outgoingGradTensor = [mpsGraph scatterNDWithUpdatesTensor:castGradTensor
                                                       indicesTensor:reshapedIndicesTensor
                                                               shape:getMPSShape(IntArrayRef(outgoing_gradient_shape))
                                                     batchDimensions:0
                                                                mode:MPSGraphScatterModeAdd
                                                                name:@"edb"];
      if (dataType == MPSDataTypeFloat16) {
        outgoingGradTensor = [mpsGraph castTensor:outgoingGradTensor toType:MPSDataTypeFloat16 name:@"castGradTensor"];
      }
      newCachedGraph->incomingGradTensor_ = incomingGradTensor;
      newCachedGraph->indicesTensor_ = indicesTensor;
      newCachedGraph->outgoingGradTensor_ = outgoingGradTensor;
    });
    auto incomingGradPlaceholder = Placeholder(cachedGraph->incomingGradTensor_, grad_);
    auto indicesPlaceholder = Placeholder(cachedGraph->indicesTensor_, indices);
    auto outgoingGradPlaceholder = Placeholder(cachedGraph->outgoingGradTensor_, outgoing_gradient);

    auto feeds = dictionaryFromPlaceholders(incomingGradPlaceholder, indicesPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outgoingGradPlaceholder);
  }
  return outgoing_gradient;
}

Tensor& masked_fill__mps(Tensor& self, const Tensor& mask, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0,
              "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
              "with ",
              value.dim(),
              " dimension(s).");
  return masked_fill__mps(self, mask, value.item());
}

Tensor& masked_scatter__mps(Tensor& self, const Tensor& mask, const Tensor& source) {
  at::assert_no_internal_overlap(self);
  TORCH_CHECK(self.scalar_type() == source.scalar_type(),
              "masked_scatter: expected self and source to have same dtypes but got",
              self.scalar_type(),
              " and ",
              source.scalar_type());

  if (self.numel() == 0) {
    return self;
  }

  TORCH_CHECK(mask.scalar_type() == ScalarType::Byte || mask.scalar_type() == ScalarType::Bool,
              "masked_scatter: expected BoolTensor or ByteTensor for mask");

  auto mask_temp =
      (mask.dim() == 0) ? c10::MaybeOwned<Tensor>::owned(mask.unsqueeze(0)) : c10::MaybeOwned<Tensor>::borrowed(mask);
  auto self_temp =
      (self.dim() == 0) ? c10::MaybeOwned<Tensor>::owned(self.unsqueeze(0)) : c10::MaybeOwned<Tensor>::borrowed(self);

  // Cannot reassign to mask_temp and self_temp here! if they are
  // owning and expand_outplace returns a borrow, the returned borrow
  // would dangle.
  auto mask_self_expanded = expand_outplace(*mask_temp, *self_temp);
  auto indices =
      at::native::expandTensors(*std::get<1>(mask_self_expanded),
                                c10::List<std::optional<at::Tensor>>({*std::move(std::get<0>(mask_self_expanded))}));
  // next broadcast all index tensors together
  try {
    indices = at::expand_outplace(indices);
  } catch (std::exception& e) {
    TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together");
  }

  if (!indices[0].has_storage() || indices[0].numel() == 0) {
    return self;
  }

  c10::List<std::optional<Tensor>> final_indices;
  final_indices.reserve(indices.size());

  for (const auto index : indices) {
    final_indices.push_back(index);
  }
  return at::index_put_out(self, *std::get<1>(mask_self_expanded), final_indices, source.resize_(indices[0].numel()));
}

Tensor& index_fill_mps_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();
  auto num_indices = index.numel();
  dim = maybe_wrap_dim(dim, self.dim());

  // Checks
  TORCH_CHECK_INDEX(index.dim() <= 1, "index_fill_(): Index is supposed to be a vector");
  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int,
              "index_fill_(): Expected dtype int32 or int64 for index");
  TORCH_CHECK(dim == 0 || dim < self.dim(), "index_fill_(): Indexing dim ", dim, " is out of bounds of tensor");

  // Empty index
  if (num_indices == 0) {
    return self;
  }

  // Scalar input
  if (self.dim() == 0 && self.numel() == 1) {
    return self.copy_(source);
  }

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* indexTensor_ = nil;
    MPSGraphTensor* updateTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  auto inputType = getMPSDataType(self);
  auto sourceType = getMPSDataType(source);
  if (inputType == MPSDataTypeUInt8 || inputType == MPSDataTypeBool) {
    inputType = MPSDataTypeInt8;
  }

  std::vector<int64_t> source_shape(self.sizes().begin(), self.sizes().end());
  source_shape[dim] = index.numel();
  auto expanded_source = source.expand(source_shape);

  @autoreleasepool {
    string key = "index_fill_mps_" + getTensorsStringKey({self, index, expanded_source}) + ":" + std::to_string(dim);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, inputType, getMPSShape(self));
      MPSGraphTensor* indexTensor = mpsGraphRankedPlaceHolder(mpsGraph, index);
      MPSGraphTensor* updateTensor = mpsGraphRankedPlaceHolder(mpsGraph, expanded_source);
      MPSGraphTensor* castedUpdateTensor = updateTensor;
      if (inputType != sourceType) {
        castedUpdateTensor = castMPSTensor(mpsGraph, updateTensor, inputType);
      }
      MPSGraphTensor* outputTensor = [mpsGraph scatterWithDataTensor:inputTensor
                                                       updatesTensor:castedUpdateTensor
                                                       indicesTensor:indexTensor
                                                                axis:(NSInteger)dim
                                                                mode:MPSGraphScatterModeSet
                                                                name:nil];
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->indexTensor_ = indexTensor;
      newCachedGraph->updateTensor_ = updateTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_,
                                              self,
                                              /*mpsShape=*/nullptr,
                                              /*gatherTensorData=*/true,
                                              /*dataType=*/inputType);
    Placeholder indexPlaceholder = Placeholder(cachedGraph->indexTensor_, index);
    Placeholder updatePlaceholder = Placeholder(cachedGraph->updateTensor_,
                                                expanded_source,
                                                /*mpsShape=*/nullptr,
                                                /*gatherTensorData=*/true,
                                                /*dataType=*/sourceType);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_,
                                                self,
                                                /*mpsShape=*/nullptr,
                                                /*gatherTensorData=*/false,
                                                /*dataType=*/inputType);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder, indexPlaceholder, updatePlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
  return self;
}

Tensor& index_fill_mps_(Tensor& self, int64_t dim, const Tensor& index, const Scalar& source) {
  return self.index_fill_(dim, index, mps::wrapped_scalar_tensor_mps(source, self.device()));
}

REGISTER_DISPATCH(index_stub, &mps::index_kernel_mps)
REGISTER_DISPATCH(index_put_stub, &mps::index_put_kernel_mps)
} // namespace at::native
