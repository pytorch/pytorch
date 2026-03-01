//  Copyright © 2022 Apple Inc.
#include <limits>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch_v2.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/Indexing.h>

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
#include <c10/util/SmallVector.h>
#include <c10/util/irange.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/native/IndexKernel.h>
#include <ATen/ops/count_nonzero.h>
#include <ATen/ops/count_nonzero_native.h>
#include <ATen/ops/embedding_dense_backward_native.h>
#include <ATen/ops/flip_native.h>
#include <ATen/ops/index.h>
#include <ATen/ops/index_add_native.h>
#include <ATen/ops/index_copy_native.h>
#include <ATen/ops/index_put.h>
#include <ATen/ops/index_reduce_native.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/masked_fill_native.h>
#include <ATen/ops/masked_scatter_native.h>
#include <ATen/ops/masked_select_native.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/nonzero_native.h>
#include <ATen/ops/ones_like.h>
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
  TORCH_CHECK(use_64bit_index || iter.can_use_32bit_indexing(),
              "kernel data offsets can't be computed using 32-bit iterator of shape ",
              iterShape);

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

static std::string getBitSizeString(const TensorBase& t) {
  return getBitSizeString(t.scalar_type());
}

static void validateInputData(const TensorIteratorBase& iter,
                              IntArrayRef index_size,
                              IntArrayRef index_stride,
                              const std::string& op) {
  const auto num_indices = index_size.size();
  TORCH_CHECK(num_indices <= 16, "Current limit allows up to 16 indices to be used in MPS indexing kernels");

  AT_ASSERT(num_indices == index_stride.size());
  AT_ASSERT(static_cast<int>(num_indices) == iter.ntensors() - 2);
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

static void dispatch_index_kernel(TensorIteratorBase& iter,
                                  IntArrayRef index_size,
                                  IntArrayRef index_stride,
                                  const std::string& kernel_name,
                                  const bool serial = false) {
  validateInputData(iter, index_size, index_stride, "index.Tensor_out");
  if (iter.numel() == 0)
    return;
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      dispatch_index_kernel(sub_iter, index_size, index_stride, kernel_name);
    }
    return;
  }
  const auto mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    const int64_t num_indices = index_size.size();
    auto indexSelectPSO = lib.getPipelineStateForFunc(kernel_name);
    auto computeEncoder = mpsStream->commandEncoder();
    size_t argumentBufferLength = sizeof(uint64_t) * num_indices;
    std::vector<uint64_t> indexAB;
    std::array<uint32_t, 4> ndim_nindiees = {static_cast<uint32_t>(iter.ndim()),
                                             static_cast<uint32_t>(index_size.size()),
                                             static_cast<uint32_t>(iter.numel()),
                                             0};
    for (uint32_t idx = 0; idx < num_indices; idx++) {
      const auto& indexTensor = iter.tensor_base(idx + 2);
      indexAB.push_back(getMTLBufferStorage(indexTensor).gpuAddress + iter_tensor_offset(iter, idx + 2));
      TORCH_CHECK(indexTensor.scalar_type() == ScalarType::Long, "index(): Expected dtype int64 for Index");
      [computeEncoder useResource:getMTLBufferStorage(indexTensor) usage:MTLResourceUsageRead];
    }
    [computeEncoder setComputePipelineState:indexSelectPSO];
    bind_iter_tensors(computeEncoder, iter, 2);
    mtl_setArgs<2>(computeEncoder,
                   indexAB,
                   iter.shape(),
                   iter.strides(0),
                   iter.strides(1),
                   iter.strides(2),
                   index_size,
                   index_stride,
                   ndim_nindiees,
                   mpsStream->getErrorBuffer());
    mtl_dispatch1DJob(computeEncoder, indexSelectPSO, serial ? 1 : iter.numel());
  });
}

static void index_kernel_mps(TensorIteratorBase& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  validateInputData(iter, index_size, index_stride, "index.Tensor_out");
  dispatch_index_kernel(
      iter, index_size, index_stride, fmt::format("index_select_{}", getBitSizeString(iter.tensor_base(0))));
}

static void index_put_kernel_mps(TensorIterator& iter,
                                 IntArrayRef index_size,
                                 IntArrayRef index_stride,
                                 bool accumulate) {
  @autoreleasepool {
    validateInputData(iter, index_size, index_stride, "index_put_impl");
    if (accumulate) {
      dispatch_index_kernel(iter,
                            index_size,
                            index_stride,
                            fmt::format("index_put_accumulate_{}", scalarToMetalTypeString(iter.tensor_base(0))));
    } else if (at::globalContext().deterministicAlgorithms()) {
      dispatch_index_kernel(iter,
                            index_size,
                            index_stride,
                            fmt::format("index_put_serial_{}", getBitSizeString(iter.tensor_base(0))),
                            true);
    } else {
      dispatch_index_kernel(
          iter, index_size, index_stride, fmt::format("index_put_{}", getBitSizeString(iter.tensor_base(0))));
    }
  }
}
} // namespace mps

TORCH_IMPL_FUNC(index_copy_out_mps)(const Tensor& self,
                                    int64_t dim,
                                    const Tensor& index,
                                    const Tensor& source,
                                    const Tensor& result) {
  using namespace mps;

  // special-case for 0-dim tensors
  if (self.dim() == 0) {
    TORCH_CHECK(index.numel() == 1,
                "index_copy_(): attempting to index a 0-dim tensor with an index tensor of size ",
                index.numel());
    int64_t idx = index.item<int64_t>();
    TORCH_CHECK(idx == 0, "index_copy_(): the only valid index for a 0-dim tensor is 0, but got ", idx);
    result.copy_(source.squeeze());
    return;
  }

  dim = maybe_wrap_dim(dim, self.dim());

  // early return for empty index
  if (index.numel() == 0) {
    result.copy_(self);
    return;
  }

  for (int64_t i = 0; i < self.dim(); i++) {
    if (i != dim) {
      TORCH_CHECK(self.size(i) == source.size(i),
                  "index_copy_(): self and source must have same size at dimension ",
                  i,
                  "; self has size ",
                  self.size(i),
                  ", source has size ",
                  source.size(i));
    }
  }

  const auto source_size_dim = source.dim() > 0 ? source.size(dim) : 1;
  TORCH_CHECK(index.numel() == source_size_dim,
              "index_copy_(): Number of indices (",
              index.numel(),
              ") should be equal to source.size(dim) (",
              source_size_dim,
              ")");

  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();

  const bool is_dense =
      self.is_contiguous() && source.is_contiguous() && result.is_contiguous() && index.is_contiguous();

  auto dense_or_strided = is_dense ? "dense" : "strided";
  auto long_or_int = (index.scalar_type() == ScalarType::Long) ? "long" : "int";
  auto indexCopyPSO = lib.getPipelineStateForFunc(
      fmt::format("index_copy_{}_{}_{}", dense_or_strided, scalarToMetalTypeString(result), long_or_int));

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      uint32_t dim_arg = static_cast<uint32_t>(dim);
      uint32_t ndim = self.dim();
      uint32_t indices_numel = index.numel();
      [computeEncoder setComputePipelineState:indexCopyPSO];
      mtl_setArgs(computeEncoder, result, self, source, index, dim_arg, self.sizes(), ndim, indices_numel);
      if (!is_dense) {
        mtl_setArgs<8>(computeEncoder, self.strides(), result.strides(), source.strides(), index.strides());
      }
      mtl_dispatch1DJob(computeEncoder, indexCopyPSO, result.numel());
    }
  });
}

static Tensor nonzero_fallback(const Tensor& self) {
  return at::nonzero(self.to("cpu")).to("mps");
}

static Tensor& nonzero_out_native_mps(const Tensor& self, Tensor& out_) {
  using namespace mps;

  int64_t nDim = self.dim();
  MPSStream* stream = getCurrentMPSStream();
  using CachedGraph = MPSUnaryCachedGraph;

  dispatch_sync_with_rethrow(stream->queue(), ^() {
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
    std::string key = "nonzero_out_native_mps" + getTensorsStringKey(self);
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
  if (self.is_complex()) {
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

  dispatch_sync_with_rethrow(stream->queue(), ^() {
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
    std::string key = "nonzero_out_native_mps" + getTensorsStringKey(self);
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
  if (self.is_complex()) {
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
    std::string key = "flip_mps:" + getTensorsStringKey({self}) + ":" + std::string([ns_dims_key UTF8String]);
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

  bool use_deterministic_algorithm = globalContext().deterministicAlgorithms();

  // TODO: Do not use deterministic algorithm for long/complex but rather implement it as Metal shader
  use_deterministic_algorithm |= source.scalar_type() == ScalarType::Long;
  use_deterministic_algorithm |= c10::isComplexType(source.scalar_type());

  if (use_deterministic_algorithm) {
    if (!result.is_same(self)) {
      result.copy_(self);
    }
    torch::List<std::optional<Tensor>> indices;
    indices.reserve(dim + 1);
    for (const auto i : c10::irange(dim)) {
      indices.emplace_back();
    }
    indices.emplace_back(index.to(at::kLong));
    const Tensor result_ = (result.dim() == 0) ? result.view(1) : result;
    const Tensor source_ = (source.dim() == 0) ? source.view(1) : source;
    result_.index_put_(indices, source_.mul(alpha), true);
    return;
  }

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
    std::string key = "index_add_mps_out" + getTensorsStringKey({self, index, source}) + ":" + std::to_string(dim);
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
  Tensor result = at::empty({0}, self.options());
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
  at::assert_no_internal_overlap(output);
  at::assert_no_overlap(output, self);
  at::assert_no_overlap(output, index);
  auto output_size = self.sizes().vec();
  if (self.dim() > 0) {
    output_size[dim] = num_indices;
  }
  at::native::resize_output(output, output_size);

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
    std::string key = "index_select_out_mps" + getTensorsStringKey({self, index}) + ":" + std::to_string(dim);
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

static inline ReductionType index_reduce_type(const std::string_view& reduce) {
  if (reduce == "prod") {
    return ReductionType::PROD;
  } else if (reduce == "mean") {
    return ReductionType::MEAN;
  } else if (reduce == "amax") {
    return ReductionType::MAX;
  } else if (reduce == "amin") {
    return ReductionType::MIN;
  } else {
    TORCH_CHECK(false, "reduce argument must be either prod, mean, amax or amin, got ", reduce, ".");
  }
}

template <typename scalar_t>
static inline scalar_t highest_value() {
  if constexpr (std::numeric_limits<scalar_t>::has_infinity) {
    return std::numeric_limits<scalar_t>::infinity();
  } else {
    return std::numeric_limits<scalar_t>::max();
  }
}

template <typename scalar_t>
static inline scalar_t lowest_value() {
  if constexpr (std::numeric_limits<scalar_t>::has_infinity) {
    return -std::numeric_limits<scalar_t>::infinity();
  } else {
    return std::numeric_limits<scalar_t>::lowest();
  }
}

template <typename scalar_t>
static inline scalar_t index_reduce_init_value(ReductionType reduction_type) {
  if (reduction_type == ReductionType::PROD) {
    return 1;
  } else if (reduction_type == ReductionType::MEAN) {
    return 0;
  } else if (reduction_type == ReductionType::MAX) {
    return lowest_value<scalar_t>();
  } else if (reduction_type == ReductionType::MIN) {
    return highest_value<scalar_t>();
  } else {
    TORCH_INTERNAL_ASSERT(false, "reduction type not supported");
  }
}

TORCH_IMPL_FUNC(index_reduce_mps_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& source,
 const std::string_view reduce,
 bool include_self,
 const Tensor& result) {
  TORCH_WARN_ONCE("index_reduce() is in beta and the API may change at any time.");
  TORCH_CHECK(self.scalar_type() != c10::kLong, "index_reduce for MPS does not support torch.long dtype");
  TORCH_CHECK(self.scalar_type() != c10::kComplexFloat, "index_reduce for MPS does not support torch.cfloat dtype");

  auto reduction_type = index_reduce_type(reduce);

  if (!result.is_same(self)) {
    result.copy_(self);
  }

  if (!include_self) {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half,
                               at::ScalarType::BFloat16,
                               result.scalar_type(),
                               "index_reduce_func_mps_exclude_input_init",
                               [&] {
                                 scalar_t init_val = index_reduce_init_value<scalar_t>(reduction_type);
                                 result.index_fill_(dim, index.to(at::ScalarType::Long), init_val);
                               });
  }

  IndexReduceParams params;
  params.index_stride = index.stride(0);
  params.reduce_dim = dim;
  params.ndim = result.dim();

  for (const auto dim : c10::irange(result.dim())) {
    params.self_strides[dim] = result.stride(dim);
    params.self_sizes[dim] = result.size(dim);
    params.source_strides[dim] = source.stride(dim);
    params.source_sizes[dim] = source.size(dim);
  }

  MPSStream* stream = getCurrentMPSStream();

  auto num_threads = source.numel();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> compute_encoder = stream->commandEncoder();
      auto pipeline_state = mps::lib.getPipelineStateForFunc(fmt::format(
          "index_reduce_{}_{}_{}", reduce, mps::scalarToMetalTypeString(result), mps::scalarToMetalTypeString(index)));
      getMPSProfiler().beginProfileKernel(pipeline_state, "index_reduce", {result, index, source});
      [compute_encoder setComputePipelineState:pipeline_state];
      mps::mtl_setArgs(compute_encoder, result, index, source, params);
      mps::mtl_dispatch1DJob(compute_encoder, pipeline_state, num_threads);
      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });

  if (reduction_type == ReductionType::MEAN) {
    auto counts = include_self ? at::ones_like(result) : at::zeros_like(result);
    counts.index_add_(dim, index, at::ones_like(source));
    counts.masked_fill_(counts.eq(0), 1);
    if (result.is_floating_point() || result.is_complex()) {
      result.div_(counts);
    } else {
      result.div_(counts, "floor");
    }
  }
}

Tensor& masked_fill__mps(Tensor& self, const Tensor& mask, const Scalar& value) {
  using namespace mps;

  if (self.numel() == 0 || mask.numel() == 0) {
    return self;
  }
  TORCH_CHECK(self.device() == mask.device(),
              "expected self and mask to be on the same device, but got mask on ",
              mask.device(),
              " and self on ",
              self.device());
  TORCH_CHECK(mask.scalar_type() == kBool, "expected mask dtype to be Bool but got ", mask.scalar_type());
  TORCH_CHECK(self.numel() <= std::numeric_limits<uint32_t>::max(),
              "masked_fill not supported for tensors of more than 2**32 elements");
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");
  auto stream = getCurrentMPSStream();
  const bool is_dense = self.is_contiguous() && b_mask->is_contiguous();
  const bool is_dense_broadcast = is_dense_broadcastable(mask, self);
  const auto flavor = is_dense ? "dense" : is_dense_broadcast ? "broadcast" : "strided";
  auto fillPSO = lib.getPipelineStateForFunc(
      fmt::format("masked_fill_scalar_{}_{}", flavor, getBitSizeString(self.scalar_type())));
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      auto mpsScalar = getMPSScalar(value, self.scalar_type());
      [computeEncoder setComputePipelineState:fillPSO];
      if (is_dense) {
        mtl_setArgs(computeEncoder, self, *b_mask, mpsScalar);
      } else if (is_dense_broadcast) {
        mtl_setArgs(computeEncoder, self, mask, mpsScalar, mask.numel());
      } else {
        mtl_setArgs(computeEncoder,
                    self,
                    *b_mask,
                    mpsScalar,
                    self.sizes(),
                    self.strides(),
                    b_mask->strides(),
                    self.ndimension());
      }
      mtl_dispatch1DJob(computeEncoder, fillPSO, self.numel());
    }
  });

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
    std::string key = "edb_mps:" + getTensorsStringKey({grad_, indices}) + ":num_weights" +
        std::to_string(num_weights) + ":padding_idx" + std::to_string(padding_idx) + ":scaled" +
        std::to_string(scale_grad_by_freq);
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

  TORCH_CHECK(indices[0].numel() <= source.numel(), "Number of elements of source < number of ones in mask");

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

  return at::index_put_out(
      self, *std::get<1>(mask_self_expanded), final_indices, source.flatten().narrow(0, 0, indices[0].numel()));
}

static void index_fill_mps_kernel(TensorIterator& iter,
                                  int64_t dim,
                                  int64_t self_dim_size,
                                  int64_t self_dim_stride,
                                  const Scalar& source) {
  if (iter.numel() == 0) {
    return;
  }
  using namespace mps;

  // Reconstruct original self from the restrided iterator output.
  // iter.tensor(0) has stride[dim]=0 and size[dim]=index_numel; restore originals.
  const Tensor& self_rs = iter.tensor(0);
  auto self_sizes = self_rs.sizes().vec();
  auto self_strides = self_rs.strides().vec();
  self_sizes[dim] = self_dim_size;
  self_strides[dim] = self_dim_stride;
  Tensor self = self_rs.as_strided(self_sizes, self_strides, self_rs.storage_offset());

  // Reconstruct original 1D index from the restrided iterator input.
  // iter.tensor(1) has shape [..., index_numel, ...] with stride[dim]=original stride.
  const Tensor& idx_rs = iter.tensor(1);
  Tensor index = idx_rs.as_strided({idx_rs.size(dim)}, {idx_rs.stride(dim)}, idx_rs.storage_offset());

  const bool is_dense = self.is_contiguous() && index.is_contiguous();
  const auto type_str = scalarToMetalTypeString(self);
  const int64_t dim_size = self_dim_size;
  const int64_t indices_numel = index.numel();

  // For large index counts, use a two-pass mask approach: mark which dim-positions
  // to fill, then iterate over all elements checking the mask. This avoids write
  // conflicts from duplicate indices and produces cache-friendly sequential writes.
  // Threshold: ≥6% fill rate (indices_numel × 16 ≥ dim_size).  Empirically this
  // is where the mask's O(numel) sequential scan beats scatter's growing write-
  // conflict overhead.
  const bool use_mask = (indices_numel * 16 >= dim_size);

  auto stream = getCurrentMPSStream();
  if (use_mask) {
    // Use at::empty (not at::zeros) to avoid a blit encoder fill that would
    // force a blit→compute encoder switch, adding latency per call.
    // The mask is zeroed via a compute kernel inside the dispatch block instead.
    // TODO: Remove me after https://github.com/pytorch/pytorch/issues/175859 is fixed
    auto mask = at::empty({dim_size}, at::TensorOptions().dtype(at::kBool).device(self.device()));
    auto zeroMaskPSO = lib.getPipelineStateForFunc("index_fill_zero_mask");
    auto setMaskPSO = lib.getPipelineStateForFunc("index_fill_set_mask");
    auto fillMaskPSO = lib.getPipelineStateForFunc(
        fmt::format("index_fill_{}_from_mask_{}", is_dense ? "dense" : "strided", type_str));

    c10::SmallVector<int64_t> all_sizes, all_strides;
    if (!is_dense) {
      for (int64_t d = 0; d < self.dim(); d++) {
        all_sizes.push_back(self.size(d));
        all_strides.push_back(self.stride(d));
      }
    }

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        auto mpsScalar = getMPSScalar(source, self.scalar_type());
        long indices_stride = index.stride(0);

        // Pass 0: zero the mask entirely within the compute encoder.
        [computeEncoder setComputePipelineState:zeroMaskPSO];
        mtl_setArgs(computeEncoder, mask);
        mtl_dispatch1DJob(computeEncoder, zeroMaskPSO, dim_size);

        [computeEncoder setComputePipelineState:setMaskPSO];
        mtl_setArgs(computeEncoder, mask, index);
        mtl_setArgs<2>(computeEncoder, dim_size, indices_stride);
        mtl_dispatch1DJob(computeEncoder, setMaskPSO, indices_numel);

        [computeEncoder setComputePipelineState:fillMaskPSO];
        mtl_setArgs(computeEncoder, self, mask);
        mtl_setBytes(computeEncoder, mpsScalar, 2);
        if (is_dense) {
          // 3D dispatch: (inner, dim, outer) avoids integer division in the kernel.
          // Compute inner_size from size products, not stride(dim): for size-1 dims,
          // is_contiguous() returns true regardless of stride, so stride(dim) can
          // exceed the actual inner element count, causing outer_size=0 (no threads).
          uint32_t dim_size_u = static_cast<uint32_t>(dim_size);
          uint32_t inner_size = 1;
          for (int64_t d = dim + 1; d < self.dim(); d++) {
            inner_size *= static_cast<uint32_t>(self.size(d));
          }
          uint32_t outer_size = static_cast<uint32_t>(self.numel() / ((int64_t)dim_size * inner_size));
          mtl_setArgs<3>(computeEncoder, dim_size_u, inner_size);
          NSUInteger maxTG = [fillMaskPSO maxTotalThreadsPerThreadgroup];
          // Fill threadgroup across dimensions to ensure adequate occupancy.
          // When inner_size is small (e.g. 1 for dim=last), use y-dim threads.
          NSUInteger tgX = std::min((NSUInteger)inner_size, maxTG);
          NSUInteger tgY = std::min((NSUInteger)dim_size_u, maxTG / tgX);
          NSUInteger tgZ = std::min((NSUInteger)outer_size, maxTG / (tgX * tgY));
          [computeEncoder dispatchThreads:MTLSizeMake(inner_size, dim_size_u, outer_size)
                    threadsPerThreadgroup:MTLSizeMake(tgX, tgY, tgZ)];
        } else {
          uint32_t dim_u = static_cast<uint32_t>(dim);
          uint32_t ndim_u = static_cast<uint32_t>(self.dim());
          mtl_setArgs<3>(computeEncoder, all_sizes, all_strides, dim_u, ndim_u);
          mtl_dispatch1DJob(computeEncoder, fillMaskPSO, self.numel());
        }
      }
    });
  } else {
    // Scatter: one thread per (index, slice-element) pair.
    const int64_t slice_numel = self.numel() / dim_size;
    auto indexFillPSO =
        lib.getPipelineStateForFunc(fmt::format("index_fill_{}_{}", is_dense ? "dense" : "strided", type_str));

    c10::SmallVector<int64_t> slice_sizes, slice_out_strides;
    if (!is_dense) {
      for (int64_t d = 0; d < self.dim(); d++) {
        if (d != dim) {
          slice_sizes.push_back(self.size(d));
          slice_out_strides.push_back(self.stride(d));
        }
      }
    }

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        auto mpsScalar = getMPSScalar(source, self.scalar_type());
        [computeEncoder setComputePipelineState:indexFillPSO];
        mtl_setArgs(computeEncoder, self, index);
        mtl_setBytes(computeEncoder, mpsScalar, 2);
        if (is_dense) {
          long dim_stride = self.stride(dim);
          mtl_setArgs<3>(computeEncoder, dim_size, dim_stride, slice_numel);
        } else {
          long dim_out_stride = self.stride(dim);
          long indices_stride = index.stride(0);
          uint32_t slice_ndim = static_cast<uint32_t>(self.dim() - 1);
          mtl_setArgs<3>(computeEncoder,
                         dim_size,
                         dim_out_stride,
                         slice_sizes,
                         slice_out_strides,
                         slice_ndim,
                         slice_numel,
                         indices_stride);
        }
        mtl_dispatch1DJob(computeEncoder, indexFillPSO, indices_numel * slice_numel);
      }
    });
  }
}

REGISTER_DISPATCH(index_stub, &mps::index_kernel_mps)
REGISTER_DISPATCH(index_fill_stub, &index_fill_mps_kernel)
REGISTER_DISPATCH(index_put_stub, &mps::index_put_kernel_mps)
} // namespace at::native
