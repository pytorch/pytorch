//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/metal/common.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/gather_native.h>
#include <ATen/ops/scatter_add_native.h>
#include <ATen/ops/scatter_native.h>
#include <ATen/ops/view_as_real.h>
#endif

namespace at::native {

namespace mps {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Scatter_metallib.h>
#endif

// Fast path applies when output, src, and index are all contiguous, src
// matches index's shape (no broadcasting), and index matches output's shape
// outside `dim`. Indexing math collapses to a single mod + div per thread.
static bool can_use_dense_scatter(const Tensor& output, const Tensor& index, int64_t dim) {
  if (!output.is_contiguous() || !index.is_contiguous()) {
    return false;
  }
  for (const auto i : c10::irange(output.dim())) {
    if (i == dim) {
      continue;
    }
    if (output.size(i) != index.size(i)) {
      return false;
    }
  }
  return true;
}

static int64_t dense_inner_size(const Tensor& output, int64_t dim) {
  int64_t inner = 1;
  for (int64_t i = dim + 1; i < output.dim(); ++i) {
    inner *= output.size(i);
  }
  return inner;
}

// Metal's [[thread_position_in_grid]] is at most a `uint`, so chunk the
// dispatch when index.numel() exceeds UINT_MAX. Each chunk reads
// `tid_offset` and adds it to its local thread index.
template <typename SetArgsFn>
static void dispatch_chunked(id<MTLComputeCommandEncoder> encoder,
                             id<MTLComputePipelineState> pso,
                             int64_t total,
                             SetArgsFn set_args) {
  const int64_t chunk_size = std::numeric_limits<uint32_t>::max();
  for (int64_t off = 0; off < total; off += chunk_size) {
    const int64_t this_chunk = std::min<int64_t>(chunk_size, total - off);
    set_args(off);
    mtl_dispatch1DJob(encoder, pso, this_chunk);
  }
}

// In-place scatter with reduce='set'. `self` already holds the output's
// initial contents (scatter_impl in TensorAdvancedIndexing.cpp does the
// self->out copy before invoking the stub). Reports out-of-bounds indices
// via the stream's async error buffer; raised on the next sync via
// MPSStream::checkLastError (matches CUDA's deferred-assert model).
static void scatter_set_metal(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int,
              "scatter: expected index to be Long or Int, got ",
              index.scalar_type());

  const auto ndim = static_cast<uint32_t>(index.dim());
  TORCH_CHECK(
      ndim <= c10::metal::max_ndim, "scatter: tensor rank ", ndim, " exceeds Metal max of ", c10::metal::max_ndim);
  const int64_t output_dim_size = self.size(dim);
  const int64_t total = index.numel();
  const bool use_dense = can_use_dense_scatter(self, index, dim) && src.is_contiguous() && src.sizes() == index.sizes();

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      if (use_dense) {
        const int64_t inner_size = dense_inner_size(self, dim);
        const int64_t index_dim_size = index.size(dim);
        auto pso = lib.getPipelineStateForFunc(
            fmt::format("scatter_set_dense_{}_{}", scalarToMetalTypeString(self), scalarToMetalTypeString(index)));
        [encoder setComputePipelineState:pso];
        dispatch_chunked(encoder, pso, total, [&](int64_t tid_offset) {
          mtl_setArgs(encoder,
                      self,
                      src,
                      index,
                      inner_size,
                      index_dim_size,
                      output_dim_size,
                      tid_offset,
                      stream->getErrorBuffer());
        });
      } else {
        auto sizes = index.sizes();
        std::array<uint32_t, 3> ndim_dim = {ndim, static_cast<uint32_t>(dim), 0};
        auto pso = lib.getPipelineStateForFunc(
            fmt::format("scatter_set_{}_{}", scalarToMetalTypeString(self), scalarToMetalTypeString(index)));
        [encoder setComputePipelineState:pso];
        dispatch_chunked(encoder, pso, total, [&](int64_t tid_offset) {
          mtl_setArgs(encoder,
                      self,
                      src,
                      index,
                      sizes,
                      self.strides(),
                      src.strides(),
                      index.strides(),
                      ndim_dim,
                      output_dim_size,
                      tid_offset,
                      stream->getErrorBuffer());
        });
      }
    }
  });
}

// Dense scatter with reduce='set' and a scalar source. Avoids the
// `at::empty + fill_` temp tensor the legacy path materialized for
// scatter.value. Falls back to allocating a src tensor and reusing
// scatter_set_metal when the dense layout assumptions don't hold.
static void scatter_fill_metal(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
  if (!can_use_dense_scatter(self, index, dim) || !index.is_contiguous()) {
    Tensor src = at::empty(index.sizes(), self.options());
    src.fill_(value);
    scatter_set_metal(self, dim, index, src);
    return;
  }

  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int,
              "scatter: expected index to be Long or Int, got ",
              index.scalar_type());
  const int64_t output_dim_size = self.size(dim);
  const int64_t inner_size = dense_inner_size(self, dim);
  const int64_t index_dim_size = index.size(dim);
  const int64_t total = index.numel();

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      // MPSScalar owns a DataPtr (non-copyable); construct inside the block so
      // ObjC's capture-by-value doesn't try to copy it. setBytes copies the
      // value into the command buffer, so the local lifetime is fine.
      MPSScalar mps_value = getMPSScalar(value, self.scalar_type());
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(
          fmt::format("scatter_set_dense_value_{}_{}", scalarToMetalTypeString(self), scalarToMetalTypeString(index)));
      [encoder setComputePipelineState:pso];
      dispatch_chunked(encoder, pso, total, [&](int64_t tid_offset) {
        mtl_setArgs(encoder,
                    self,
                    mps_value,
                    index,
                    inner_size,
                    index_dim_size,
                    output_dim_size,
                    tid_offset,
                    stream->getErrorBuffer());
      });
    }
  });
}

// XOR every element's sign bit. Used as the pre- and post-pass that brackets
// a signed int64 amin/amax scatter (Metal's atomic_min/max work on ulong,
// so we encode signed values by flipping the sign bit to make signed order
// match unsigned order; the encoding is its own inverse).
static void dispatch_signbit_xor_long(id<MTLComputeCommandEncoder> encoder, MPSStream* stream, const Tensor& self) {
  TORCH_CHECK(self.is_contiguous(), "scatter_reduce(amin/amax) on int64 requires contiguous self");
  auto pso = lib.getPipelineStateForFunc("scatter_signbit_xor_long");
  [encoder setComputePipelineState:pso];
  dispatch_chunked(encoder, pso, self.numel(), [&](int64_t tid_offset) { mtl_setArgs(encoder, self, tid_offset); });
}

// Atomic Metal scatter for one of {add, prod, amin, amax}. Picks the dense
// kernel when output/src/index are contiguous and shape-aligned outside dim,
// strided kernel otherwise.
static void scatter_reduce_metal(const Tensor& self,
                                 int64_t dim,
                                 const Tensor& index,
                                 const Tensor& src,
                                 const char* op) {
  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int,
              "scatter_reduce: expected index to be Long or Int, got ",
              index.scalar_type());
  const auto ndim = static_cast<uint32_t>(index.dim());
  TORCH_CHECK(
      ndim <= c10::metal::max_ndim, "scatter: tensor rank ", ndim, " exceeds Metal max of ", c10::metal::max_ndim);
  const int64_t output_dim_size = self.size(dim);
  const int64_t total = index.numel();
  const bool use_dense = can_use_dense_scatter(self, index, dim) && src.is_contiguous() && src.sizes() == index.sizes();
  // Signed int64 amin/amax needs an encode/decode bracket so signed ordering
  // maps onto the unsigned atomic_min/max Metal exposes. Requires contiguous
  // self so we can sweep it as a flat ulong buffer. The ulong atomic_min/max
  // intrinsic is only available at runtime on macOS 15+.
  const bool needs_signbit_xor =
      self.scalar_type() == ScalarType::Long && (std::string_view(op) == "amin" || std::string_view(op) == "amax");
  if (needs_signbit_xor) {
    TORCH_CHECK(is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS),
                "scatter_reduce(amin/amax) on int64 requires macOS 15 or newer");
    TORCH_CHECK(self.is_contiguous(), "scatter_reduce(amin/amax) on int64 currently requires contiguous self");
  }

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      if (needs_signbit_xor) {
        dispatch_signbit_xor_long(encoder, stream, self);
      }
      if (use_dense) {
        const int64_t inner_size = dense_inner_size(self, dim);
        const int64_t index_dim_size = index.size(dim);
        auto pso = lib.getPipelineStateForFunc(
            fmt::format("scatter_{}_dense_{}_{}", op, scalarToMetalTypeString(self), scalarToMetalTypeString(index)));
        [encoder setComputePipelineState:pso];
        dispatch_chunked(encoder, pso, total, [&](int64_t tid_offset) {
          mtl_setArgs(encoder,
                      self,
                      src,
                      index,
                      inner_size,
                      index_dim_size,
                      output_dim_size,
                      tid_offset,
                      stream->getErrorBuffer());
        });
      } else {
        auto sizes = index.sizes();
        std::array<uint32_t, 3> ndim_dim = {ndim, static_cast<uint32_t>(dim), 0};
        auto pso = lib.getPipelineStateForFunc(
            fmt::format("scatter_{}_strided_{}_{}", op, scalarToMetalTypeString(self), scalarToMetalTypeString(index)));
        [encoder setComputePipelineState:pso];
        dispatch_chunked(encoder, pso, total, [&](int64_t tid_offset) {
          mtl_setArgs(encoder,
                      self,
                      src,
                      index,
                      sizes,
                      self.strides(),
                      src.strides(),
                      index.strides(),
                      ndim_dim,
                      output_dim_size,
                      tid_offset,
                      stream->getErrorBuffer());
        });
      }
      if (needs_signbit_xor) {
        dispatch_signbit_xor_long(encoder, stream, self);
      }
    }
  });
}
} // namespace mps

static Tensor maybe_expand_0_dim(const Tensor& t) {
  return t.dim() == 0 ? t.view({1}) : t;
}

static Tensor expand_index_as_real(const Tensor& index) {
  auto index_view = maybe_expand_0_dim(index);
  std::vector<int64_t> index_expanded_sizes = index_view.sizes().vec();
  index_expanded_sizes.push_back(2);
  auto index_expanded = index_view.unsqueeze(-1).expand(index_expanded_sizes);
  return index_expanded;
}

TORCH_IMPL_FUNC(gather_out_mps)
(const Tensor& self_arg, int64_t dim, const Tensor& index, bool sparse_grad, const Tensor& output) {
  using namespace mps;

  if (self_arg.numel() == 0 || index.numel() == 0) {
    return;
  }
  auto self = self_arg.dim() == 0 ? self_arg.view({1}) : self_arg;
  dim = at::maybe_wrap_dim(dim, self.dim());

  TORCH_CHECK(!sparse_grad, "sparse_grad not supported in MPS yet")
  TORCH_CHECK(self.scalar_type() == output.scalar_type(), "gather(): self and output must have the same scalar type");
  TORCH_CHECK(dim >= 0 && dim < self.dim(), "gather(): Indexing dim ", dim, " is out of bounds of tensor");

  if (self.is_complex()) {
    auto self_real = at::view_as_real(self);
    auto index_expanded = expand_index_as_real(index);
    auto output_real = at::view_as_real(maybe_expand_0_dim(output));
    structured_gather_out_mps::impl(self_real, dim, index_expanded, sparse_grad, output_real);
    return;
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* indexTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  @autoreleasepool {
    MPSShape* input_shape = getMPSShape(self);
    MPSShape* index_shape = getMPSShape(index);
    uint32_t num_input_dims = [input_shape count];
    uint32_t num_index_dims = [index_shape count];
    TORCH_CHECK(num_input_dims == num_index_dims, "Input and index must have same rank")

    // Determine if we need to slice into the input tensor
    bool needSlice = false;

    for (const auto i : c10::irange(num_input_dims)) {
      TORCH_CHECK(i == dim || [index_shape[i] intValue] <= [input_shape[i] intValue],
                  "Index dim must not exceed input dim except at gathering axis")
      if (i != dim && [index_shape[i] intValue] < [input_shape[i] intValue])
        needSlice = true;
    }
    auto input_type = getMPSDataType(self);
    auto output_type = getMPSDataType(output);
    if (input_type == MPSDataTypeUInt8) {
      input_type = MPSDataTypeInt8;
    }
    if (output_type == MPSDataTypeUInt8) {
      output_type = MPSDataTypeInt8;
    }
    std::string key = "gather_out_mps" + getTensorsStringKey({self, index, output}) + ":" + std::to_string(dim);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_type, getMPSShape(self));
      MPSGraphTensor* indexTensor = mpsGraphRankedPlaceHolder(mpsGraph, index);
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->indexTensor_ = indexTensor;

      MPSGraphTensor* getInput = inputTensor;

      // Slice into the input tensor IF NEEDED
      if (needSlice) {
        NSMutableArray<NSNumber*>* starts = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
        NSMutableArray<NSNumber*>* ends = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
        NSMutableArray<NSNumber*>* strides = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];

        for (const auto i : c10::irange(num_input_dims)) {
          // All strides are 1
          strides[i] = @1;
          // All starts are 0
          starts[i] = @0;
          ends[i] = (i != dim) ? index_shape[i] : input_shape[i];
        }

        getInput = [mpsGraph sliceTensor:inputTensor starts:starts ends:ends strides:strides name:nil];
      }

      // PyTorch issue #135240: MPS reshape underneath goes haywire in
      // gatherAlongAxis before MacOS 15.2 if it has multiple dimensions but can
      // be squeezed into 1D. We need to squeeze out the extra dims pre 15.2
      bool workaroundSingleDim = (self_arg.squeeze().sizes().size() == 1 && self_arg.sizes().size() > 1);
      bool isMacos15_2 = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_2_PLUS);
      if (workaroundSingleDim and !isMacos15_2) {
        const int64_t dims = self_arg.sizes().size();
        int64_t size = self_arg.squeeze().sizes()[0];
        auto shape = [[NSMutableArray alloc] initWithCapacity:dims];
        for (int i = 0; i < dims; ++i) {
          [shape addObject:[NSNumber numberWithInt:size]];
        }
        getInput = [mpsGraph broadcastTensor:getInput toShape:shape name:nil];
        indexTensor = [mpsGraph broadcastTensor:indexTensor toShape:shape name:nil];
      }
      MPSGraphTensor* castIndexTensor = [mpsGraph castTensor:indexTensor
                                                      toType:MPSDataTypeInt32
                                                        name:(NSString* _Nonnull)nil];

      C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wobjc-method-access")
      MPSGraphTensor* outputTensor = [mpsGraph gatherAlongAxis:(NSInteger)dim
                                             withUpdatesTensor:getInput
                                                 indicesTensor:castIndexTensor
                                                          name:nil];
      C10_DIAGNOSTIC_POP()

      if (workaroundSingleDim) {
        outputTensor = [mpsGraph sliceTensor:outputTensor dimension:0 start:0 length:output.sizes()[0] name:nil];
      }
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, input_shape, true, input_type);
    Placeholder indexPlaceholder = Placeholder(cachedGraph->indexTensor_, index, index_shape);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output, nullptr, false, output_type);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder, indexPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

static const char* reduce_op_to_mps_string(const ReductionType& op) {
  switch (op) {
    case ReductionType::SUM:
      return "add";
    case ReductionType::PROD:
      return "prod";
    case ReductionType::MIN:
      return "amin";
    case ReductionType::MAX:
      return "amax";
    case ReductionType::MEAN:
      // Pre-existing behavior: scatter_reduce(mean) on MPS accumulated as
      // sum (no divide-by-count). True mean semantics is follow-up work.
      return "add";
  }
  TORCH_CHECK(false, "Unsupported reduction type: ", static_cast<int>(op));
}

// Common pre-dispatch shape/dtype checks shared by all reduce-mode stubs.
static void scatter_reduce_check(const Tensor& self, const Tensor& index, const Tensor& src) {
  TORCH_CHECK(self.scalar_type() == src.scalar_type(), "scatter(): self and src must have the same scalar type");
  TORCH_CHECK(index.dim() == self.dim() && index.dim() == src.dim(), "Input, index and src must have same rank");
}

// scatter_stub: in-place set with a Tensor src. self is the writable output
// (pre-loaded by scatter_impl in TensorAdvancedIndexing.cpp).
static void scatter_mps_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  if (self.numel() == 0 || index.numel() == 0 || src.numel() == 0) {
    return;
  }
  TORCH_CHECK(self.scalar_type() == src.scalar_type(), "scatter(): self and src must have the same scalar type");
  // Complex types map to float2/half2 in the Metal kernels (see
  // scalarToMetalTypeString), so no view_as_real shim is needed.
  auto self_view = self.dim() == 0 ? self.view({1}) : self;
  auto src_view = maybe_expand_0_dim(src);
  auto index_view = maybe_expand_0_dim(index);
  dim = at::maybe_wrap_dim(dim, self_view.dim());
  TORCH_CHECK(dim >= 0 && dim < self_view.dim(), "scatter(): Indexing dim ", dim, " is out of bounds of tensor");
  TORCH_CHECK(index_view.dim() == self_view.dim() && index_view.dim() == src_view.dim(),
              "Input, index and src must have same rank");
  for (const auto i : c10::irange(self_view.dim())) {
    TORCH_CHECK(index_view.size(i) <= src_view.size(i), "Index dim must not exceed src dim");
    TORCH_CHECK(i == dim || index_view.size(i) <= self_view.size(i),
                "Index dim must not exceed input dim except at gathering axis");
  }
  mps::scatter_set_metal(self_view, dim, index_view, src_view);
}

// scatter_fill_stub: in-place set with a scalar value. Uses the dense
// scatter_set_dense_value kernel when possible (skips temp-src materialization).
static void scatter_fill_mps_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
  if (self.numel() == 0 || index.numel() == 0) {
    return;
  }
  // MPSScalar carries c10::complex<float>/<Half> via its union and Metal kernels
  // are templated on float2/half2, so complex scalars flow through unchanged.
  auto self_view = self.dim() == 0 ? self.view({1}) : self;
  auto index_view = maybe_expand_0_dim(index);
  dim = at::maybe_wrap_dim(dim, self_view.dim());
  TORCH_CHECK(dim >= 0 && dim < self_view.dim(), "scatter(): Indexing dim ", dim, " is out of bounds of tensor");
  TORCH_CHECK(index_view.dim() == self_view.dim(), "Input and index must have same rank");
  for (const auto i : c10::irange(self_view.dim())) {
    TORCH_CHECK(i == dim || index_view.size(i) <= self_view.size(i),
                "Index dim must not exceed input dim except at gathering axis");
  }
  mps::scatter_fill_metal(self_view, dim, index_view, value);
}

// Shared scatter-reduce dispatch used by add / reduce / scalar_reduce /
// reduce_two stubs. mean accumulates as sum here; the divide-by-count
// post-pass for scatter_reduce.two happens in shared scatter_reduce_two.
static void scatter_reduce_dispatch(const Tensor& self,
                                    int64_t dim,
                                    const Tensor& index,
                                    const Tensor& src,
                                    const ReductionType& reduce) {
  if (self.numel() == 0 || index.numel() == 0 || src.numel() == 0) {
    return;
  }
  // Match CPU: scatter_reduce(mean) isn't defined for bool.
  TORCH_CHECK(reduce != ReductionType::MEAN || self.scalar_type() != ScalarType::Bool,
              "scatter_reduce: reduce='mean' not implemented for 'Bool'");
  auto self_view = self.dim() == 0 ? self.view({1}) : self;
  auto src_view = maybe_expand_0_dim(src);
  auto index_view = maybe_expand_0_dim(index);
  dim = at::maybe_wrap_dim(dim, self_view.dim());
  scatter_reduce_check(self_view, index_view, src_view);
  mps::scatter_reduce_metal(self_view, dim, index_view, src_view, reduce_op_to_mps_string(reduce));
}

static void scatter_add_mps_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  scatter_reduce_dispatch(self, dim, index, src, ReductionType::SUM);
}

static void scatter_reduce_mps_kernel(const Tensor& self,
                                      int64_t dim,
                                      const Tensor& index,
                                      const Tensor& src,
                                      const ReductionType& reduce) {
  scatter_reduce_dispatch(self, dim, index, src, reduce);
}

static void scatter_scalar_reduce_mps_kernel(const Tensor& self,
                                             int64_t dim,
                                             const Tensor& index,
                                             const Scalar& value,
                                             const ReductionType& reduce) {
  Tensor src = at::empty(index.sizes(), self.options());
  src.fill_(value);
  scatter_reduce_dispatch(self, dim, index, src, reduce);
}

static void scatter_reduce_two_mps_kernel(const Tensor& self,
                                          const int64_t dim,
                                          const Tensor& index,
                                          const Tensor& src,
                                          const ReductionType& reduce) {
  scatter_reduce_dispatch(self, dim, index, src, reduce);
}

REGISTER_DISPATCH(scatter_stub, &scatter_mps_kernel);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_mps_kernel);
REGISTER_DISPATCH(scatter_add_stub, &scatter_add_mps_kernel);
REGISTER_DISPATCH(scatter_reduce_stub, &scatter_reduce_mps_kernel);
REGISTER_DISPATCH(scatter_scalar_reduce_stub, &scatter_scalar_reduce_mps_kernel);
REGISTER_DISPATCH(scatter_reduce_two_stub, &scatter_reduce_two_mps_kernel);
} // namespace at::native
