//  Copyright © 2023 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/SortingUtils.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/metal/common.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/as_strided.h>
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

// 2D (n_rows, sort_size) view of self for last-dim sort, nullopt if copy required.
static std::optional<Tensor> try_view_as_2d_lastdim(const Tensor& self, int sort_size) {
  int64_t n_rows = self.numel() / sort_size;
  std::vector<int64_t> new_shape = {n_rows, sort_size};
  auto maybe_strides = at::detail::computeStride(self.sizes(), self.strides(), new_shape);
  if (!maybe_strides.has_value()) {
    return std::nullopt;
  }
  return at::as_strided(self, new_shape, *maybe_strides);
}

static int select_tptg(int sort_size, size_t elem_size, int n_rows) {
  constexpr int ILP = static_cast<int>(c10::metal::ILP_PER_THREAD);
  int potential_tptg = at::ceil_div(sort_size, ILP);
  // Few rows: shrink TPTG to force multi-block, giving the GPU more TGs.
  if (n_rows <= 2 && sort_size > 2048) {
    constexpr int target_total_tgs = 4;
    int target_blocks_per_row = std::max(1, target_total_tgs / std::max(n_rows, 1));
    int target_elems_per_tg = std::max(2048, at::ceil_div(sort_size, target_blocks_per_row));
    int target_tptg = target_elems_per_tg / ILP;
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
  const int tptg1024_n_blocks = at::ceil_div(sort_size, 1024 * ILP);
  if (tptg == 1024 && tptg1024_n_blocks > 1 && n_rows >= 8) {
    tptg = 512;
  }
  return tptg;
}

static bool should_use_radix(int sort_size, size_t elem_size, int elems_per_tg) {
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

static void sort_radix(const Tensor& input,
                       const Tensor& values,
                       const Tensor& indices,
                       int64_t dim,
                       bool descending,
                       int sort_size) {
  bool need_permute = (dim != input.ndimension() - 1);
  Tensor work_in = input;
  if (need_permute) {
    work_in = input.movedim(dim, -1).contiguous();
    sort_size = work_in.size(work_in.ndimension() - 1);
  }

  int n_rows = static_cast<int>(work_in.numel() / sort_size);
  // 16/32-bit types use 8-bit radix (256 bins) - halves global memory traffic
  // vs 4-bit radix while keeping the same total block-local work
  // (RBITS sub-passes x n_passes = total bits either way). 8-bit types stay on
  // 4-bit radix since 2 passes is already the minimum meaningful pass count.
  const size_t elem_size = values.element_size();
  const int radix_bits = (elem_size == 1) ? 4 : 8;
  const int radix_size = 1 << radix_bits;
  const int n_passes = (8 * static_cast<int>(elem_size)) / radix_bits;
  // 2-byte many-row: drop RTPTG 1024->512 to halve per-TG tgmem -> 2 TGs/core.
  const bool small_tg = (elem_size == 2 && n_rows >= 32);
  const int RADIX_TPTG = (elem_size == 2 && !small_tg) ? 1024 : 512;
  const int radix_ept = (elem_size == 1) ? 8 : 4;
  const int RADIX_ELEMS_PER_TG = RADIX_TPTG * radix_ept;
  int n_blocks = at::ceil_div(sort_size, RADIX_ELEMS_PER_TG);
  int n_entries = radix_size * n_blocks;

  auto opts_val = values.options();
  auto opts_u32 = at::TensorOptions().dtype(at::kInt).device(values.device());
  const bool direct_final_write = !need_permute;
  // Use int16 storage reinterpreted as ushort in the kernel: halves the
  // intermediate index memory traffic. Only valid when global indices fit in
  // 16 bits (sort_size <= 65536). The kernel never treats these as signed.
  const bool use_u16 = direct_final_write && sort_size <= 65536;
  auto opts_idx = use_u16 ? at::TensorOptions().dtype(at::kShort).device(values.device()) : opts_u32;

  auto work_in_view = work_in.contiguous().reshape({n_rows, sort_size});
  auto keys_0 = at::empty({n_rows, sort_size}, opts_val);
  auto keys_1 = at::empty({n_rows, sort_size}, opts_val);
  auto idxs_0 = at::empty({n_rows, sort_size}, opts_idx);
  auto idxs_1 = at::empty({n_rows, sort_size}, opts_idx);
  auto histograms = at::empty({n_rows, n_entries}, opts_u32);

  const auto type_str = scalarToMetalTypeString(values);
  const char* tptg_suffix = small_tg ? "_tptg512" : "";
  const char* u16_suffix = use_u16 ? "_u16" : "";
  const auto count_kernel = fmt::format("radix_count_{}_{}bit{}", type_str, radix_bits, tptg_suffix);

  constexpr int kMaxFusedBlocks = 4;
  constexpr int kFusedWorkCap = 128;
  const bool use_fused_count_scan = n_blocks <= kMaxFusedBlocks && !small_tg;
  const bool use_fused_scan =
      !use_fused_count_scan && (n_blocks <= kMaxFusedBlocks) && (n_blocks * n_blocks * n_rows <= kFusedWorkCap);
  const char* fused_prefix = use_fused_scan ? "fused_" : "";
  const auto scatter_kernel =
      fmt::format("radix_scatter_{}{}_{}bit{}{}", fused_prefix, type_str, radix_bits, tptg_suffix, u16_suffix);
  const auto scatter_final_kernel =
      fmt::format("radix_scatter_{}final_{}_{}bit{}{}", fused_prefix, type_str, radix_bits, tptg_suffix, u16_suffix);
  const auto count_scan_kernel =
      use_fused_count_scan ? fmt::format("radix_count_scan_{}_{}bit_mb4", type_str, radix_bits) : std::string{};

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto enc = mpsStream->commandEncoder();
      auto count_pso = use_fused_count_scan ? nil : lib.getPipelineStateForFunc(count_kernel);
      auto count_scan_pso = use_fused_count_scan ? lib.getPipelineStateForFunc(count_scan_kernel) : nil;
      MTLComputePipelineState_t scan_pso = nil;
      if (!use_fused_scan && !use_fused_count_scan)
        scan_pso = lib.getPipelineStateForFunc("radix_scan");
      auto scatter_pso = lib.getPipelineStateForFunc(scatter_kernel);
      MTLComputePipelineState_t scatter_final_pso =
          direct_final_write ? lib.getPipelineStateForFunc(scatter_final_kernel) : nil;

      bool ping = false;
      for (int pass = 0; pass < n_passes; pass++) {
        int shift = pass * radix_bits;
        bool first_pass = (pass == 0);
        bool last_pass = (pass == n_passes - 1);
        bool use_direct = direct_final_write && last_pass;

        const Tensor& k_in = first_pass ? work_in_view : (ping ? keys_1 : keys_0);
        const Tensor& i_in = ping ? idxs_1 : idxs_0;
        const Tensor& k_out_buf = ping ? keys_0 : keys_1;
        const Tensor& i_out_buf = ping ? idxs_0 : idxs_1;

        const auto dims = std::array<int32_t, 3>{sort_size, n_blocks, shift};
        const auto flags = std::array<uint8_t, 2>{static_cast<uint8_t>(descending), static_cast<uint8_t>(first_pass)};

        if (use_fused_count_scan) {
          getMPSProfiler().beginProfileKernel(count_scan_pso, count_scan_kernel, {k_in});
          [enc setComputePipelineState:count_scan_pso];
          mtl_setArgs(enc, k_in, histograms, dims, descending);
          [enc dispatchThreadgroups:MTLSizeMake(1, n_rows, 1) threadsPerThreadgroup:MTLSizeMake(RADIX_TPTG, 1, 1)];
          getMPSProfiler().endProfileKernel(count_scan_pso);
        } else {
          getMPSProfiler().beginProfileKernel(count_pso, count_kernel, {k_in});
          [enc setComputePipelineState:count_pso];
          mtl_setArgs(enc, k_in, histograms, dims, descending);
          [enc dispatchThreadgroups:MTLSizeMake(n_blocks, n_rows, 1)
              threadsPerThreadgroup:MTLSizeMake(RADIX_TPTG, 1, 1)];
          getMPSProfiler().endProfileKernel(count_pso);

          if (!use_fused_scan) {
            getMPSProfiler().beginProfileKernel(scan_pso, "radix_scan", {histograms});
            [enc setComputePipelineState:scan_pso];
            mtl_setArgs(enc, histograms, n_entries);
            [enc dispatchThreadgroups:MTLSizeMake(1, n_rows, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
            getMPSProfiler().endProfileKernel(scan_pso);
          }
        }

        auto pso = use_direct ? scatter_final_pso : scatter_pso;
        const auto& kernel_name = use_direct ? scatter_final_kernel : scatter_kernel;
        getMPSProfiler().beginProfileKernel(pso, kernel_name, {k_in});
        [enc setComputePipelineState:pso];
        if (use_direct) {
          mtl_setArgs(enc, k_in, i_in, values, indices, histograms, dims, flags);
        } else {
          mtl_setArgs(enc, k_in, i_in, k_out_buf, i_out_buf, histograms, dims, flags);
        }
        [enc dispatchThreadgroups:MTLSizeMake(n_blocks, n_rows, 1) threadsPerThreadgroup:MTLSizeMake(RADIX_TPTG, 1, 1)];
        getMPSProfiler().endProfileKernel(pso);

        ping = !ping;
      }
    }
  });

  if (direct_final_write) {
    // Already written directly to values/indices in the last scatter.
    return;
  }

  bool ping = (n_passes % 2 == 1);
  const Tensor& final_keys = ping ? keys_1 : keys_0;
  const Tensor& final_idxs = ping ? idxs_1 : idxs_0;
  values.copy_(final_keys.view(work_in.sizes()).movedim(-1, dim));
  indices.copy_(final_idxs.view(work_in.sizes()).movedim(-1, dim));
}

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

static void sort_multi_block(const Tensor& input,
                             const Tensor& values,
                             const Tensor& indices,
                             int64_t dim,
                             bool descending,
                             int sort_size,
                             int64_t stride_sort,
                             int64_t stride_seg,
                             int tptg,
                             bool stable) {
  const int elems_per_tg = tptg * static_cast<int>(c10::metal::ILP_PER_THREAD);
  const int n_rows = static_cast<int>(input.numel() / sort_size);
  const int n_blocks = at::ceil_div(sort_size, elems_per_tg);
  const bool need_permute = (dim != input.ndimension() - 1);

  Tensor work_in = input;
  if (need_permute) {
    work_in = input.movedim(dim, -1).contiguous();
    stride_sort = 1;
    stride_seg = sort_size;
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
  const char* stable_sfx = stable ? "_stable" : "";
  // mb_sort_block has stable variants; mb_merge doesn't (stable on stable input).
  const auto block_fn = fmt::format("mb_sort_block_{}_tptg{}{}{}", type_str, tptg, u16_sfx, stable_sfx);
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
      mtl_setArgs(enc, work_in, buf_v0, buf_i0, sort_size, std::array<int64_t, 2>{stride_sort, stride_seg}, descending);
      [enc dispatchThreadgroups:MTLSizeMake(n_blocks, n_rows, 1) threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
      getMPSProfiler().endProfileKernel(block_pso);

      // Stage 2: pairwise merge passes, doubling run length each round
      if (n_blocks > 1) {
        auto merge_pso = lib.getPipelineStateForFunc(merge_fn);
        auto final_pso = direct_final ? lib.getPipelineStateForFunc(final_fn) : nil;
        bool ping = false;
        for (int merge_tiles = 2; (merge_tiles / 2) < n_blocks; merge_tiles *= 2) {
          const bool is_last = direct_final && (merge_tiles >= n_blocks);
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
    values.copy_(final_v.view(work_in.sizes()).movedim(-1, dim));
    indices.copy_(final_i.view(work_in.sizes()).movedim(-1, dim));
  } else {
    values.copy_(final_v.view(values.sizes()));
    indices.copy_(final_i.view(indices.sizes()));
  }
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

  Tensor out_vals = values;
  Tensor out_inds = indices;
  const bool need_copy_back = !values.is_contiguous() || !indices.is_contiguous();
  if (need_copy_back) {
    out_vals = at::empty(self.sizes(), values.options());
    out_inds = at::empty(self.sizes(), indices.options());
  }

  const int n_rows = static_cast<int>(self.numel() / sort_size);
  const int tptg = select_tptg(sort_size, self.element_size(), n_rows);
  const int elems_per_tg = tptg * static_cast<int>(c10::metal::ILP_PER_THREAD);
  const bool is_last_dim = (dim == self.ndimension() - 1);
  const bool stable_kernel = stable.value_or(false);

  const bool use_radix = should_use_radix(sort_size, self.element_size(), elems_per_tg);

  if (use_radix) {
    sort_radix(self, out_vals, out_inds, dim, descending, sort_size);
  } else {
    // For last-dim sort, try a strided view to skip .contiguous(); the kernels
    // read with (stride_sort, stride_seg) so any view that flattens to 2D works.
    // For non-last-dim, sort_multi_block handles the permute+contiguous itself.
    Tensor input = self;
    int64_t input_dim = dim;
    int64_t stride_sort = 1;
    int64_t stride_seg = sort_size;
    if (is_last_dim) {
      auto strided = try_view_as_2d_lastdim(self, sort_size);
      if (strided.has_value()) {
        input = *strided;
        stride_sort = input.stride(1);
        stride_seg = input.stride(0);
      } else {
        input = self.contiguous();
      }
      // After the view, the sort dim is the last dim of `input` (which may be 2D
      // even when self is 1D or N-D). For the contiguous path it's the same as dim.
      input_dim = input.ndimension() - 1;
    }

    if (is_last_dim && sort_size <= elems_per_tg) {
      sort_single_block(input, out_vals, out_inds, descending, sort_size, stride_sort, stride_seg, tptg, stable_kernel);
    } else {
      sort_multi_block(
          input, out_vals, out_inds, input_dim, descending, sort_size, stride_sort, stride_seg, tptg, stable_kernel);
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
