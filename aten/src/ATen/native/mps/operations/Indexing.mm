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
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Pool.h>
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
#include <ATen/ops/nonzero_static_native.h>
#include <ATen/ops/ones_like.h>
#endif

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
      // Metal atomic-add is non-associative for floating/complex types, so
      // duplicate indices race on the result. Integer adds are associative
      // and remain deterministic
      const auto dtype = iter.tensor_base(0).scalar_type();
      if (at::isFloatingType(dtype) || at::isComplexType(dtype)) {
        at::globalContext().alertNotDeterministic("index_put_with_accumulate_mps");
      }
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

  // Base copy: non-indexed slices come straight from self. Skipped for in-place
  // index_copy_, where result already aliases self.
  if (!result.is_same(self)) {
    result.copy_(self);
  }

  const auto is_dense = source.is_contiguous() && result.is_contiguous() && index.is_contiguous();
  const auto indices_numel = index.numel();
  const auto slice_numel = source.numel() / indices_numel;
  if (slice_numel == 0) {
    return;
  }

  const auto use_32 = canUse32BitIndexMath(result) && canUse32BitIndexMath(source) && canUse32BitIndexMath(index);
  auto dense_or_strided = is_dense ? "dense" : "strided";
  auto long_or_int = (index.scalar_type() == ScalarType::Long) ? "long" : "int";
  auto indexCopyPSO = lib.getPipelineStateForFunc(fmt::format(
      "index_copy_{}_{}_{}_{}", dense_or_strided, scalarToMetalTypeString(result), long_or_int, use_32 ? "32" : "64"));

  const auto dim_size = result.size(dim);
  c10::SmallVector<int64_t> slice_sizes, slice_out_strides, slice_source_strides;
  if (!is_dense) {
    slice_sizes.reserve(result.dim() - 1);
    slice_out_strides.reserve(result.dim() - 1);
    slice_source_strides.reserve(result.dim() - 1);
    for (int64_t d = 0; d < result.dim(); d++) {
      if (d != dim) {
        slice_sizes.push_back(result.size(d));
        slice_out_strides.push_back(result.stride(d));
        slice_source_strides.push_back(source.stride(d));
      }
    }
  }

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:indexCopyPSO];
      mtl_setArgs(computeEncoder, result, source, index);
      if (is_dense) {
        const auto inner = result.stride(dim);
        const auto outer = slice_numel / inner;
        mtl_setArgs<3>(computeEncoder, dim_size, inner, indices_numel);
        auto maxTG = [indexCopyPSO maxTotalThreadsPerThreadgroup];
        auto tgX = std::min<NSUInteger>(inner, maxTG);
        auto tgY = std::min<NSUInteger>(indices_numel, std::max<NSUInteger>(1, maxTG / tgX));
        auto tgZ = std::min<NSUInteger>(outer, std::max<NSUInteger>(1, maxTG / (tgX * tgY)));
        [computeEncoder dispatchThreads:MTLSizeMake(inner, indices_numel, outer)
                  threadsPerThreadgroup:MTLSizeMake(tgX, tgY, tgZ)];
      } else {
        auto dim_out_stride = result.stride(dim);
        auto dim_source_stride = source.stride(dim);
        auto indices_stride = index.stride(0);
        auto slice_ndim = static_cast<uint32_t>(result.dim() - 1);
        mtl_setArgs<3>(computeEncoder,
                       dim_size,
                       dim_out_stride,
                       dim_source_stride,
                       slice_sizes,
                       slice_out_strides,
                       slice_source_strides,
                       slice_ndim,
                       slice_numel,
                       indices_stride);
        mtl_dispatch1DJob(computeEncoder, indexCopyPSO, indices_numel * slice_numel);
      }
    }
  });
}

// Metal kernel-based nonzero using prefix-sum + scatter.
// Step 1: Per-element exclusive prefix sum of nonzero flags + block totals.
// Step 2: GPU prefix sum of block totals → block offsets + total count.
// Host (optional):   Read back total count, allocate output, unless max_element is provided
// Step 3: Scatter multi-dimensional indices into the output.
static void nonzero_impl_mps(const Tensor& self, Tensor& out_, std::optional<int64_t> max_elements) {
  using namespace mps;

  TORCH_CHECK(self.numel() < std::numeric_limits<int>::max(),
              "nonzero is not supported for tensors with more than INT_MAX elements, "
              "See https://github.com/pytorch/pytorch/issues/51871");
  TORCH_CHECK(out_.dtype() == at::kLong, "Expected output type to be Long, but got ", out_.dtype());
  TORCH_CHECK(self.device() == out_.device(),
              "expected self and out to be on the same device, but got out on ",
              out_.device(),
              " and self on ",
              self.device());
  TORCH_CHECK(out_.is_mps());

  Tensor input = self.contiguous();
  const int64_t nDim = self.dim();
  const auto numel = static_cast<uint32_t>(input.numel());
  const auto type_str = scalarToMetalTypeString(input);
  MPSStream* stream = getCurrentMPSStream();

  auto pso_step1 = lib.getPipelineStateForFunc(fmt::format("count_nonzero_prefix_sum_{}", type_str));
  auto pso_step2 = lib.getPipelineStateForFunc("prefix_sum_blocks");
  auto pso_step3 = lib.getPipelineStateForFunc(fmt::format("scatter_nonzero_indices_{}", type_str));
  TORCH_INTERNAL_ASSERT([pso_step1 maxTotalThreadsPerThreadgroup] == [pso_step3 maxTotalThreadsPerThreadgroup],
                        "nonzero: step 1 and step 3 threadgroup sizes must match");

  uint32_t threads_per_group = static_cast<uint32_t>([pso_step1 maxTotalThreadsPerThreadgroup]);
  uint32_t num_blocks = (numel + threads_per_group - 1) / threads_per_group;

  auto tmp = at::empty({input.numel() + 2 * num_blocks + 1}, input.options().dtype(kInt));
  Tensor prefix_buf = tmp.slice(0, 0, numel);
  Tensor block_sums_buf = tmp.slice(0, numel, numel + num_blocks);
  Tensor block_offsets_buf = tmp.slice(0, numel + num_blocks, numel + 2 * num_blocks);
  Tensor total_nonzero_buf = tmp.slice(0, numel + 2 * num_blocks, numel + 2 * num_blocks + 1);

  // Steps 1+2: compute prefix sums and block offsets entirely on GPU
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();

      [computeEncoder setComputePipelineState:pso_step1];
      mtl_setArgs(computeEncoder, input, prefix_buf, block_sums_buf);
      mtl_dispatch1DJob(computeEncoder, pso_step1, numel);

      [computeEncoder setComputePipelineState:pso_step2];
      mtl_setArgs(computeEncoder, block_sums_buf, block_offsets_buf, total_nonzero_buf, num_blocks);
      uint32_t tg_size_blocks = std::min(1024u, c10::metal::round_up(num_blocks, 32u));
      [computeEncoder dispatchThreads:MTLSizeMake(tg_size_blocks, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size_blocks, 1, 1)];
    }
  });

  if (!max_elements) {
    // Dynamic path: sync to learn output size
    const int64_t total_nonzero = total_nonzero_buf.item<int>();
    at::native::resize_output(out_, {total_nonzero, nDim});
    max_elements = total_nonzero;
  }

  if (out_.numel() == 0) {
    return;
  }

  bool contiguous_output = out_.is_contiguous();
  Tensor out = contiguous_output ? out_ : at::empty_like(out_, MemoryFormat::Contiguous);

  int ndim_int = static_cast<int>(nDim);
  int max_entries = static_cast<int>(*max_elements);

  // Step 3: scatter indices, capped at max_entries
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:pso_step3];
      mtl_setArgs(computeEncoder, input, prefix_buf, out, ndim_int, input.sizes(), block_offsets_buf, max_entries);
      mtl_dispatch1DJob(computeEncoder, pso_step3, numel);
    }
  });

  if (!contiguous_output) {
    out_.copy_(out);
  }
}

Tensor& nonzero_out_mps(const Tensor& self, Tensor& out_) {
  int64_t nDim = self.dim();
  if (self.numel() == 0) {
    at::native::resize_output(out_, {0, nDim});
    return out_;
  }

  nonzero_impl_mps(self, out_, std::nullopt);
  return out_;
}

Tensor nonzero_mps(const Tensor& self) {
  Tensor out = at::empty({0}, self.options().dtype(kLong));
  return nonzero_out_mps(self, out);
}

Tensor& nonzero_static_out_mps(const Tensor& self, int64_t size, int64_t fill_value, Tensor& result) {
  TORCH_CHECK(size >= 0, "nonzero_static: 'size' must be an non-negative integer");

  int64_t nDim = self.dim();
  if (result.dim() != 2 || result.size(0) != size || result.size(1) != nDim) {
    at::native::resize_output(result, {size, nDim});
  }

  if (result.size(0) == 0 || result.size(1) == 0) {
    return result;
  }

  result.fill_(fill_value);

  if (self.numel() == 0) {
    return result;
  }

  nonzero_impl_mps(self, result, size);
  return result;
}

Tensor nonzero_static_mps(const Tensor& self, int64_t size, int64_t fill_value) {
  TORCH_CHECK(size >= 0, "nonzero_static: 'size' must be an non-negative integer");
  int64_t nDim = self.dim();
  auto result = at::empty({size, nDim}, at::TensorOptions().dtype(kLong).device(kMPS));
  nonzero_static_out_mps(self, size, fill_value, result);
  return result;
}

Tensor masked_select_mps(const Tensor& self, const Tensor& mask) {
  Tensor result = at::empty({0}, self.options());
  return mps::masked_select_out_mps_impl(result, self, mask);
}

Tensor& masked_select_out_mps(const Tensor& self, const Tensor& mask, Tensor& result) {
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

// Validate index in [0, dim_size) once (one thread per index) on the given
// encoder, so the following gather/scatter kernel can clamp instead of
// branch-and-report per element. Out-of-bounds surfaces as an async
// AcceleratorError on the next stream sync.
static void encodeIndexBoundsCheck(id<MTLComputeCommandEncoder> encoder,
                                   at::mps::MPSStream* stream,
                                   const Tensor& index,
                                   int64_t dim_size) {
  using namespace mps;
  auto pso = lib.getPipelineStateForFunc(fmt::format("index_check_bounds_{}", scalarToMetalTypeString(index)));
  [encoder setComputePipelineState:pso];
  mtl_setArgs(encoder, index);
  mtl_setArgs<1>(encoder, static_cast<uint32_t>(index.stride(0)), dim_size, stream->getErrorBuffer());
  mtl_dispatch1DJob(encoder, pso, index.numel());
}

TORCH_IMPL_FUNC(index_add_mps_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& source,
 const Scalar& alpha,
 const Tensor& result) {
  using namespace mps;
  dim = maybe_wrap_dim(dim, self.dim());

  // Structured out variant: result is a distinct tensor that must start as self.
  if (!result.is_same(self)) {
    result.copy_(self);
  }
  if (index.numel() == 0 || source.numel() == 0) {
    return;
  }

  // Floating-point atomic add is non-associative, so the Metal kernel's
  // accumulation order is non-deterministic. Fall back to index_put_ with
  // accumulate=true when deterministic algorithms are requested.
  if (globalContext().deterministicAlgorithms()) {
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

  const Tensor result_ = (result.dim() == 0) ? result.view(1) : result;
  const Tensor source_ = (source.dim() == 0) ? source.view(1) : source;
  const Tensor index_ = (index.dim() == 0) ? index.view(1) : index;

  // Empty indexed dim: CPU index_add early-returns on an empty self without
  // range-checking, so match that (no error) and skip the scatter, whose clamp
  // to self_sizes[dim] - 1 == -1 would otherwise compute an invalid offset.
  if (result_.size(dim) == 0) {
    return;
  }

  // fp16/bf16/chalf atomic add is emulated with a compare-and-swap loop that
  // collapses under index contention, while fp32/cfloat have a native atomic
  // add. Upcast the accumulation only when many indices map into few slots:
  // the measured crossover is ~16 indices per slot (M4 Max), below which the
  // in-place low-precision atomic is faster and avoids the cast plus the extra
  // fp32 buffers. Upcasting also matches the pre-Metal cast-to-float numerics.
  const auto contention = index_.numel() / std::max<int64_t>(1, result_.size(dim));
  ScalarType acc_type = result_.scalar_type();
  if (contention >= 16) {
    if (acc_type == kHalf || acc_type == kBFloat16) {
      acc_type = kFloat;
    } else if (acc_type == kComplexHalf) {
      acc_type = kComplexFloat;
    }
  }
  const bool needs_acc_cast = acc_type != result_.scalar_type();
  const Tensor acc_result = needs_acc_cast ? result_.to(acc_type) : result_;
  const Tensor acc_source = needs_acc_cast ? source_.to(acc_type) : source_;

  IndexReduceParams params;
  params.index_stride = index_.stride(0);
  params.reduce_dim = dim;
  params.ndim = acc_result.dim();
  for (const auto d : c10::irange(acc_result.dim())) {
    params.self_strides[d] = acc_result.stride(d);
    params.self_sizes[d] = acc_result.size(d);
    params.source_strides[d] = acc_source.stride(d);
    params.source_sizes[d] = acc_source.size(d);
  }

  MPSStream* stream = getCurrentMPSStream();
  auto num_threads = acc_source.numel();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();
      encodeIndexBoundsCheck(computeEncoder, stream, index_, acc_result.size(dim));
      auto pipeline_state = lib.getPipelineStateForFunc(
          fmt::format("index_add_{}_{}", scalarToMetalTypeString(acc_result), scalarToMetalTypeString(index_)));
      getMPSProfiler().beginProfileKernel(pipeline_state, "index_add", {acc_result, index_, acc_source});
      [computeEncoder setComputePipelineState:pipeline_state];
      mtl_setArgs(computeEncoder, acc_result, index_, acc_source, params);
      mtl_setBytes(computeEncoder, getMPSScalar(alpha, acc_type), 4);
      mtl_dispatch1DJob(computeEncoder, pipeline_state, num_threads);
      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });
  if (needs_acc_cast) {
    result_.copy_(acc_result);
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

  // Empty index: nothing to gather and no indices to range-check.
  if (num_indices == 0) {
    return output;
  }

  const Tensor index_ = (index.dim() == 0) ? index.view(1) : index;

  // Empty input/output (a non-indexed dim is 0): there is nothing to gather, but
  // still range-check the indices against self.size(dim) to match CPU/CUDA --
  // indexing into a 0-sized dim is out of bounds. Skips dispatching a 0-sized
  // gather grid.
  if (self.numel() == 0) {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();
        encodeIndexBoundsCheck(computeEncoder, stream, index_, self.size(dim));
      }
    });
    return output;
  }

  // Scalar input
  if (self.dim() == 0 && self.numel() == 1) {
    output.copy_(self);
    return output;
  }

  // Fast path: contiguous tensors viewed as [outer, dim, inner], gathered with a
  // 3D grid so each thread copies one element with no coordinate decomposition.
  // Compute inner from size products (not stride(dim)): for size-1 dims a tensor
  // is contiguous regardless of stride, so stride(dim) can overcount.
  if (self.is_contiguous() && output.is_contiguous() && index_.is_contiguous()) {
    uint32_t inner = 1;
    for (const auto d : c10::irange(dim + 1, self.dim())) {
      inner *= static_cast<uint32_t>(self.size(d));
    }
    const uint32_t in_dim_size = self.size(dim);
    const auto outer = self.numel() / (static_cast<int64_t>(in_dim_size) * inner);

    // A gathered slice (inner contiguous elements) is copied verbatim, so widen
    // the copy unit to the largest power-of-two byte size dividing the slice span
    // (and both base offsets) to maximize memory throughput, esp. for fp16.
    const uint32_t elem_size = output.element_size();
    const uint64_t row_bytes = static_cast<uint64_t>(inner) * elem_size;
    const uint64_t self_off_bytes = static_cast<uint64_t>(self.storage_offset()) * elem_size;
    const uint64_t out_off_bytes = static_cast<uint64_t>(output.storage_offset()) * elem_size;
    uint32_t copy_bytes = 8;
    while (copy_bytes > elem_size &&
           ((row_bytes % copy_bytes) || (self_off_bytes % copy_bytes) || (out_off_bytes % copy_bytes))) {
      copy_bytes /= 2;
    }
    const uint32_t inner_units = static_cast<uint32_t>(row_bytes / copy_bytes);

    IndexSelectParams params;
    params.inner = inner_units;
    params.in_dim_size = in_dim_size;
    params.out_dim_size = static_cast<uint32_t>(num_indices);
    params.index_stride = static_cast<uint32_t>(index_.stride(0));

    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();
        encodeIndexBoundsCheck(computeEncoder, stream, index_, self.size(dim));
        auto pipeline_state = lib.getPipelineStateForFunc(
            fmt::format("index_select_dim_dense_{}bit_{}", copy_bytes * 8, scalarToMetalTypeString(index_)));
        getMPSProfiler().beginProfileKernel(pipeline_state, "index_select", {self, index_});
        [computeEncoder setComputePipelineState:pipeline_state];
        mtl_setArgs(computeEncoder, output, index_, self, params);
        const MTLSize grid = MTLSizeMake(inner_units, num_indices, outer);
        const NSUInteger maxTG = [pipeline_state maxTotalThreadsPerThreadgroup];
        const NSUInteger tgX = std::min<NSUInteger>(inner_units, maxTG);
        const NSUInteger tgY = std::min<NSUInteger>(num_indices, std::max<NSUInteger>(1, maxTG / tgX));
        const NSUInteger tgZ = std::min<NSUInteger>(outer, std::max<NSUInteger>(1, maxTG / (tgX * tgY)));
        [computeEncoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tgX, tgY, tgZ)];
        getMPSProfiler().endProfileKernel(pipeline_state);
      }
    });
    return output;
  }

  // Strided fallback: one thread per output element, offsets from strides.
  // params.source_* describe the output (iterated), params.self_* the input.
  IndexReduceParams params;
  params.index_stride = index_.stride(0);
  params.reduce_dim = dim;
  params.ndim = output.dim();
  for (const auto d : c10::irange(output.dim())) {
    params.source_strides[d] = output.stride(d);
    params.source_sizes[d] = output.size(d);
    params.self_strides[d] = self.stride(d);
    params.self_sizes[d] = self.size(d);
  }

  auto num_threads = output.numel();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();
      encodeIndexBoundsCheck(computeEncoder, stream, index_, self.size(dim));
      auto pipeline_state = lib.getPipelineStateForFunc(
          fmt::format("index_select_dim_{}_{}", getBitSizeString(output), scalarToMetalTypeString(index_)));
      getMPSProfiler().beginProfileKernel(pipeline_state, "index_select", {self, index_});
      [computeEncoder setComputePipelineState:pipeline_state];
      mtl_setArgs(computeEncoder, output, index_, self, params);
      mtl_dispatch1DJob(computeEncoder, pipeline_state, num_threads);
      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });

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
  // Atomic prod/mean are non-associative for floating-point; alert unless we're
  // on an order-invariant reduction (amin/amax) or an integer dtype. Mirrors
  // CUDA's index_reduce_func_cuda_impl alert, narrowed to dtypes/ops that
  // actually produce non-deterministic output on MPS.
  const auto dtype = self.scalar_type();
  if ((reduction_type == ReductionType::PROD || reduction_type == ReductionType::MEAN) &&
      (at::isFloatingType(dtype) || at::isComplexType(dtype))) {
    at::globalContext().alertNotDeterministic("index_reduce_mps");
  }

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

  return self;
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

  bool was_scalar = self.dim() == 0;

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

  at::index_put_out(
      self, *std::get<1>(mask_self_expanded), final_indices, source.flatten().narrow(0, 0, indices[0].numel()));
  if (was_scalar) {
    self.squeeze_();
  }
  return self;
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
