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

static int select_tptg(int sort_size, size_t elem_size, int n_rows) {
  int potential_tptg = at::ceil_div(sort_size, TN);
  // Few rows: shrink TPTG to force multi-block, giving the GPU more TGs.
  if (n_rows <= 2 && sort_size > 2048) {
    constexpr int target_total_tgs = 4;
    int target_blocks_per_row = std::max(1, target_total_tgs / std::max(n_rows, 1));
    int target_elems_per_tg = std::max(2048, at::ceil_div(sort_size, target_blocks_per_row));
    int target_tptg = target_elems_per_tg / TN;
    potential_tptg = std::min(potential_tptg, target_tptg);
  }

  int tptg = std::clamp<int>(std::bit_ceil(static_cast<unsigned>(std::max(potential_tptg, 1))), 32, 1024);

  // 8-byte types: tgmem stages 8 bytes (value) + 4 bytes (uint index) per
  // element, i.e. ELEMS_PER_TG * 12 bytes. At TPTG=1024 that's 48 KB, over
  // the 32 KB tgmem budget. Clamp to 256 (12 KB, leaves room for 2 TGs/core)
  if (elem_size > 4) {
    tptg = std::min(tptg, 256);
  }
  // TPTG=1024 uses ~24KB tgmem which limits occupancy, drop to 512 (~12KB) when rows hide the extra merge
  const int tptg1024_n_blocks = at::ceil_div(sort_size, 1024 * TN);
  if (tptg == 1024 && tptg1024_n_blocks > 1 && n_rows >= 8) {
    tptg = 512;
  }
  return tptg;
}

// returns true when multi-block merge sort would need so many merge-dispatch
// passes that MPSGraph's sort is expected to win
static bool should_use_mpsgraph_fallback(int sort_size, size_t elem_size, int elems_per_tg) {
  if (elem_size > 4)
    return false;
  int n_blocks_merge = at::ceil_div(sort_size, elems_per_tg);
  int merge_rounds = 0;
  for (int m = 2; (m / 2) < n_blocks_merge; m *= 2)
    merge_rounds++;
  int merge_dispatches = 1 + merge_rounds;
  int n_radix_passes = (elem_size <= 1) ? 2 : (elem_size <= 2) ? 2 : 4;
  int radix_dispatches = n_radix_passes * 3;
  return radix_dispatches <= 2 * merge_dispatches + 2;
}

static void sort_mpsgraph_fallback(const Tensor& self,
                                   const Tensor& values,
                                   const Tensor& indices,
                                   int64_t dim,
                                   bool descending) {
  values.copy_(self);
  MPSStream* stream = getCurrentMPSStream();
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *selfTensor = nil, *valuesTensor = nil, *indicesTensor = nil;
  };
  @autoreleasepool {
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
    Placeholder valuesPlaceholder = Placeholder(cachedGraph->valuesTensor, values);
    Placeholder indicesPlaceholder = Placeholder(cachedGraph->indicesTensor, indices);
    auto feeds = dictionaryFromPlaceholders(inputPlaceholder);
    auto results = dictionaryFromPlaceholders(valuesPlaceholder, indicesPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}

static void sort_single_block(const Tensor& input,
                              const Tensor& values,
                              const Tensor& indices,
                              bool descending,
                              int sort_size,
                              int64_t stride_sort,
                              int64_t stride_seg,
                              int tptg) {
  auto n_rows = static_cast<int>(input.numel() / sort_size);
  const auto kernel = fmt::format("sort_block_{}_tptg{}", scalarToMetalTypeString(input), tptg);

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

static void sort_multi_block(const Tensor& input,
                             const Tensor& values,
                             const Tensor& indices,
                             int64_t dim,
                             bool descending,
                             int sort_size,
                             int tptg) {
  const int elems_per_tg = tptg * TN;
  const int n_rows = static_cast<int>(input.numel() / sort_size);
  const int n_blocks = at::ceil_div(sort_size, elems_per_tg);
  const bool need_permute = (dim != input.ndimension() - 1);

  Tensor work_in = input;
  std::vector<int64_t> inv_perm;
  if (need_permute) {
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < input.ndimension(); i++)
      if (i != dim)
        perm.push_back(i);
    perm.push_back(dim);
    work_in = input.permute(perm).contiguous();
    inv_perm.resize(perm.size());
    for (size_t i = 0; i < perm.size(); i++)
      inv_perm[perm[i]] = static_cast<int64_t>(i);
  }

  // ushort indices halve merge bandwidth; direct_final lets the last merge
  // write long indices straight to output, skipping a widen copy.
  const bool direct_final = !need_permute;
  const bool use_u16 = direct_final && sort_size <= 65536;
  auto opts_val = values.options();
  auto opts_idx = use_u16 ? at::TensorOptions().dtype(at::kShort).device(values.device())
                          : at::TensorOptions().dtype(at::kInt).device(values.device());

  auto buf_v0 = at::empty({n_rows, sort_size}, opts_val);
  auto buf_i0 = at::empty({n_rows, sort_size}, opts_idx);
  auto buf_v1 = n_blocks > 1 ? at::empty({n_rows, sort_size}, opts_val) : Tensor{};
  auto buf_i1 = n_blocks > 1 ? at::empty({n_rows, sort_size}, opts_idx) : Tensor{};

  const auto type_str = scalarToMetalTypeString(values);
  const char* u16_sfx = use_u16 ? "_u16" : "";
  const auto block_fn = fmt::format("mb_sort_block_{}_tptg{}{}", type_str, tptg, u16_sfx);
  const auto merge_fn = fmt::format("mb_merge_{}_tptg{}{}", type_str, tptg, u16_sfx);
  const auto final_fn = fmt::format("mb_merge_final_{}_tptg{}{}", type_str, tptg, u16_sfx);

  int total_rounds = 0;
  for (int m = 2; (m / 2) < n_blocks; m *= 2)
    ++total_rounds;

  auto mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto enc = mpsStream->commandEncoder();

      // Stage 1: independently sort each block
      auto block_pso = lib.getPipelineStateForFunc(block_fn);
      getMPSProfiler().beginProfileKernel(block_pso, block_fn, {work_in});
      [enc setComputePipelineState:block_pso];
      mtl_setArgs(enc,
                  work_in,
                  buf_v0,
                  buf_i0,
                  sort_size,
                  std::array<int64_t, 2>{1, static_cast<int64_t>(sort_size)},
                  descending);
      [enc dispatchThreadgroups:MTLSizeMake(n_blocks, n_rows, 1) threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
      getMPSProfiler().endProfileKernel(block_pso);

      // Stage 2: pairwise merge passes, doubling run length each round
      if (n_blocks > 1) {
        auto merge_pso = lib.getPipelineStateForFunc(merge_fn);
        auto final_pso = direct_final ? lib.getPipelineStateForFunc(final_fn) : nil;
        bool ping = false;
        int round = 0;
        for (int merge_tiles = 2; (merge_tiles / 2) < n_blocks; merge_tiles *= 2, ++round) {
          const bool is_last = direct_final && (round == total_rounds - 1);
          const auto& v_in = ping ? buf_v1 : buf_v0;
          const auto& i_in = ping ? buf_i1 : buf_i0;
          const auto& v_out = is_last ? values : (ping ? buf_v0 : buf_v1);
          const auto& i_out = is_last ? indices : (ping ? buf_i0 : buf_i1);
          auto pso = is_last ? final_pso : merge_pso;
          ping = !ping;

          getMPSProfiler().beginProfileKernel(pso, is_last ? final_fn : merge_fn, {v_in});
          [enc setComputePipelineState:pso];
          mtl_setArgs(
              enc, v_in, i_in, v_out, i_out, std::array<int32_t, 3>{sort_size, merge_tiles, n_blocks}, descending);
          [enc dispatchThreadgroups:MTLSizeMake(n_blocks, n_rows, 1) threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
          getMPSProfiler().endProfileKernel(pso);
        }
      }
    }
  });

  // direct_final already wrote to values/indices.
  if (direct_final && n_blocks > 1)
    return;

  const auto& final_v = (total_rounds % 2 == 1) ? buf_v1 : buf_v0;
  const auto& final_i = (total_rounds % 2 == 1) ? buf_i1 : buf_i0;
  if (need_permute) {
    values.copy_(final_v.view(work_in.sizes()).permute(inv_perm));
    indices.copy_(final_i.view(work_in.sizes()).permute(inv_perm));
  } else {
    values.copy_(final_v.view(values.sizes()));
    indices.copy_(final_i.view(indices.sizes()));
  }
}

void kthvalue_out_mps_impl(const Tensor& self, int64_t k, int64_t dim, Tensor& values, Tensor& indices) {
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

  Tensor out_vals = values;
  Tensor out_inds = indices;
  const bool need_copy_back = !values.is_contiguous() || !indices.is_contiguous();
  if (need_copy_back) {
    out_vals = at::empty(self.sizes(), values.options());
    out_inds = at::empty(self.sizes(), indices.options());
  }

  const int n_rows = static_cast<int>(self.numel() / sort_size);
  const int tptg = select_tptg(sort_size, self.element_size(), n_rows);
  const int elems_per_tg = tptg * TN;
  const bool is_last_dim = (dim == self.ndimension() - 1);

  const bool use_mpsgraph = should_use_mpsgraph_fallback(sort_size, self.element_size(), elems_per_tg);

  if (use_mpsgraph) {
    sort_mpsgraph_fallback(self, out_vals, out_inds, dim, descending);
  } else {
    Tensor input = self.contiguous();
    if (is_last_dim && sort_size <= elems_per_tg) {
      sort_single_block(
          input, out_vals, out_inds, descending, sort_size, /*stride_sort=*/1, /*stride_seg=*/sort_size, tptg);
    } else {
      sort_multi_block(input, out_vals, out_inds, dim, descending, sort_size, tptg);
    }
  }

  if (need_copy_back) {
    values.copy_(out_vals);
    indices.copy_(out_inds);
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
