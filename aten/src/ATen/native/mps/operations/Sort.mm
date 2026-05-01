//  Copyright © 2023 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/SortingUtils.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/kthvalue_native.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/sort_native.h>
#endif
namespace at::native {
namespace {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Sort_metallib.h>
#endif

// TODO: reuse DEFAULT_ILP from c10/metal/common.h
static constexpr int TN = 4; // elements per thread

static int select_tptg(int sort_size, size_t elem_size) {
  int potential_tptg = at::ceil_div(sort_size, TN);
  int tptg = std::clamp<int>(std::bit_ceil(static_cast<unsigned>(std::max(potential_tptg, 1))), 32, 1024);

  // 8-byte types: tgmem stages 8 bytes (value) + 4 bytes (uint index) per
  // element, i.e. ELEMS_PER_TG * 12 bytes. At TPTG=1024 that's 48 KB, over
  // the 32 KB tgmem budget. Clamp to 256 (12 KB, leaves room for 2 TGs/core)
  if (elem_size > 4) {
    tptg = std::min(tptg, 256);
  }
  return tptg;
}

// Single-block sort (last-dim only). Loads the segment into threadgroup
// memory, sorts in place, and writes indices directly.
static void sort_single_block(const Tensor& input,
                              const Tensor& values,
                              const Tensor& indices,
                              bool descending,
                              int sort_size,
                              int64_t stride_sort,
                              int64_t stride_seg,
                              int tptg,
                              bool stable) {
  auto n_rows = static_cast<int>(input.numel() / sort_size);
  const char* stable_sfx = stable ? "_stable" : "";
  const auto kernel = fmt::format("sort_block_{}_tptg{}{}", scalarToMetalTypeString(input), tptg, stable_sfx);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto enc = mpsStream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(kernel);
      getMPSProfiler().beginProfileKernel(pso, kernel, {input});
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, input, values, indices, sort_size, std::array<int64_t, 2>{stride_sort, stride_seg}, descending);
      [enc dispatchThreadgroups:MTLSizeMake(1, n_rows, 1) threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

void kthvalue_out_mps_impl(const Tensor& self, int64_t k, int64_t dim, Tensor& values, Tensor& indices) {
  using namespace mps;
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return;
  }
  // Handle empty tensors
  if (self.numel() == 0) {
    values.copy_(self);
    indices.copy_(values.toType(at::ScalarType::Long));
    return;
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(self.scalar_type()), "kthvalue is not implemented for complex types");
  // issue #154890, raising error to prevent crash within MPSGraph until
  // workaround is implemented.
  TORCH_CHECK(self.dim() - dim <= 4, "On-going issue on MPSGraph topk when ndims() - axis > 4, see issue #154890");

  auto stream = getCurrentMPSStream();
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *selfTensor = nil, *valuesTensor = nil, *indicesTensor = nil;
  };

  // MPSGraph kthvalue is always sorted.
  @autoreleasepool {
    // Input as placeholders
    MPSShape* input_shape = getMPSShape(self);
    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];
    std::string key = std::string("kthvalue:") + [ns_shape_key UTF8String] + ":" + getMPSTypeString(self) + ":k" +
        std::to_string(k) + ":dim" + std::to_string(dim);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), input_shape);

      MPSGraphTensor* castInputTensor = newCachedGraph->selfTensor;
      MPSDataType dataType = getMPSDataType(self);
      // #issue 104398441 sortWithTensor and argsortWithTensor
      if (dataType != MPSDataTypeInt32 && dataType != MPSDataTypeFloat32 && dataType != MPSDataTypeFloat16) {
        dataType = (dataType & MPSDataTypeFloatBit) ? MPSDataTypeFloat32 : MPSDataTypeInt32;
        castInputTensor = [mpsGraph castTensor:newCachedGraph->selfTensor toType:dataType name:@"castInputTensor"];
      }
      MPSGraphTensor* sortedTensor = [mpsGraph sortWithTensor:castInputTensor
                                                         axis:(NSUInteger)dim
                                                   descending:false
                                                         name:nil];
      sortedTensor = [mpsGraph sliceTensor:sortedTensor
                                 dimension:(NSUInteger)dim
                                     start:((NSUInteger)k - 1)
                                    length:1
                                      name:nil];
      MPSGraphTensor* argSortedTensor = [mpsGraph argSortWithTensor:castInputTensor
                                                               axis:(NSInteger)dim
                                                         descending:false
                                                               name:@"kthvalue_out"];
      argSortedTensor = [mpsGraph sliceTensor:argSortedTensor
                                    dimension:dim
                                        start:((NSUInteger)k - 1)
                                       length:1
                                         name:nil];
      newCachedGraph->valuesTensor = sortedTensor;
      newCachedGraph->indicesTensor = argSortedTensor;
    });
    Placeholder inputPlaceholder = Placeholder(cachedGraph->selfTensor, self);
    // Outputs as placeholders
    Placeholder valuesPlaceholder = Placeholder(cachedGraph->valuesTensor, values);
    Placeholder indicesPlaceholder = Placeholder(cachedGraph->indicesTensor, indices);
    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    auto results = dictionaryFromPlaceholders(valuesPlaceholder, indicesPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}
} // anonymous namespace

// sort
TORCH_IMPL_FUNC(sort_stable_out_mps)
(const Tensor& self,
 std::optional<bool> stable,
 int64_t dim,
 bool descending,
 const Tensor& values,
 const Tensor& indices) {
  using namespace mps;

  if (self.numel() == 0) {
    return;
  }

  dim = maybe_wrap_dim(dim, self.dim(), true);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return;
  }

  int sort_size = static_cast<int>(self.size(dim));
  if (sort_size <= 1) {
    values.copy_(self);
    indices.zero_();
    return;
  }

  // Single-block sort: last dim, segment fits in one threadgroup, and at
  // least two rows so the dispatch uses enough GPU cores to beat MPSGraph.
  // Everything else falls through to MPSGraph.
  const bool is_last_dim = (dim == self.ndimension() - 1);
  if (is_last_dim) {
    const int n_rows = static_cast<int>(self.numel() / sort_size);
    const int tptg = select_tptg(sort_size, self.element_size());
    const int elems_per_tg = tptg * TN;
    if (n_rows >= 2 && sort_size <= elems_per_tg) {
      Tensor input = self.contiguous();
      Tensor out_vals = values;
      Tensor out_inds = indices;
      const bool need_copy_back = !values.is_contiguous() || !indices.is_contiguous();
      if (need_copy_back) {
        out_vals = at::empty(self.sizes(), values.options());
        out_inds = at::empty(self.sizes(), indices.options());
      }
      sort_single_block(input,
                        out_vals,
                        out_inds,
                        descending,
                        sort_size,
                        /*stride_sort=*/1,
                        /*stride_seg=*/sort_size,
                        tptg,
                        /*stable=*/stable.value_or(false));
      if (need_copy_back) {
        values.copy_(out_vals);
        indices.copy_(out_inds);
      }
      return;
    }
  }

  // MPSGraph fallback for everything else.
  values.copy_(self);
  MPSStream* stream = getCurrentMPSStream();
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *selfTensor = nil, *valuesTensor = nil, *indicesTensor = nil;
  };
  @autoreleasepool {
    // Input as placeholders
    MPSShape* input_shape = getMPSShape(self);
    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];
    std::string key = std::string("sort:") + [ns_shape_key UTF8String] + ":" + getMPSTypeString(self) + ":dim" +
        std::to_string(dim) + ":descending" + std::to_string(descending);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), input_shape);

      MPSGraphTensor* castInputTensor = castToIHFTypes(mpsGraph, newCachedGraph->selfTensor, self);
      MPSGraphTensor* sortedTensor = [mpsGraph sortWithTensor:castInputTensor
                                                         axis:(NSInteger)dim
                                                   descending:(BOOL)descending
                                                         name:@"sort_out"];
      if ([sortedTensor dataType] != getMPSDataType(values)) {
        sortedTensor = castMPSTensor(mpsGraph, sortedTensor, values.scalar_type());
      }
      MPSGraphTensor* argSortedTensor = [mpsGraph argSortWithTensor:castInputTensor
                                                               axis:(NSInteger)dim
                                                         descending:(BOOL)descending
                                                               name:@"argsort_out"];
      if ([argSortedTensor dataType] != getMPSDataType(indices)) {
        argSortedTensor = castMPSTensor(mpsGraph, argSortedTensor, indices.scalar_type());
      }
      newCachedGraph->valuesTensor = sortedTensor;
      newCachedGraph->indicesTensor = argSortedTensor;
    });
    Placeholder inputPlaceholder = Placeholder(cachedGraph->selfTensor, self);
    // Outputs as placeholders
    Placeholder valuesPlaceholder = Placeholder(cachedGraph->valuesTensor, values);
    Placeholder indicesPlaceholder = Placeholder(cachedGraph->indicesTensor, indices);
    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    auto results = dictionaryFromPlaceholders(valuesPlaceholder, indicesPlaceholder);

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}

std::tuple<Tensor&, Tensor&> kthvalue_out_mps(const Tensor& self,
                                              int64_t k,
                                              int64_t dim_,
                                              bool keepdim,
                                              Tensor& values,
                                              Tensor& indices) {
  // See note [Writing Nondeterministic Operations]
  // If there are duplicate elements of the kth value, the procedure for choosing which
  // of the duplicates to use for the indices output is nondeterministic.
  at::globalContext().alertNotDeterministic("kthvalue MPS");

  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  int64_t slicesize = self.dim() == 0 ? 1 : self.size(dim);
  TORCH_CHECK(k >= 1 && k <= slicesize, "kthvalue(): selected number k out of range for dimension ", dim);
  at::assert_no_overlap(self, values);
  _reduction_with_indices_allocate_or_resize_output(values, indices, self, dim, keepdim);

  kthvalue_out_mps_impl(self, k, dim, values, indices);

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }

  return std::forward_as_tuple(values, indices);
}
} // namespace at::native
