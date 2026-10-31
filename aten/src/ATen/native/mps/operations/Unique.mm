//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIndexing.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_unique2_native.h>
#include <ATen/ops/_unique_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/argsort.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/unique_consecutive_native.h>
#include <ATen/ops/unique_dim_consecutive_native.h>
#include <ATen/ops/unique_dim_native.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Unique_metallib.h>
#endif

// Unique rows of the contiguous (rows, cols) `self`; the flat ops pass a single column.
static std::tuple<Tensor, Tensor, Tensor> _unique_rows_mps(const Tensor& self,
                                                           const bool return_inverse,
                                                           const bool return_counts,
                                                           const bool consecutive) {
  using namespace mps;

  const auto rows = self.size(0), cols = self.size(1);

  // Step 1: sort (unless consecutive).
  TORCH_CHECK_NOT_IMPLEMENTED(consecutive || rows <= std::numeric_limits<int32_t>::max(),
                              "unique: sorting more than 2^31 - 1 elements is not supported on MPS");
  Tensor sorted_values = self;
  Tensor sort_idx;
  if (!consecutive && cols == 1) {
    std::tie(sorted_values, sort_idx) = at::sort(self.view(-1));
  } else if (!consecutive) {
    sort_idx = arange(rows, self.options().dtype(kLong));
    for (auto c = cols - 1; c >= 0; --c) {
      // index_select misreads strided inputs spanning more than 2^32 elements.
      const auto keys = self.select(1, c).index({sort_idx});
      sort_idx = sort_idx.index_select(0, argsort(keys, /*stable=*/true));
    }
    sorted_values = self.index_select(0, sort_idx);
  }

  // Step 2: boundary mask (int32).
  constexpr int64_t max_threads = std::numeric_limits<uint32_t>::max();
  // Wide rows are split across up to max_row_threads threads, each covering at least cols_per_thread columns.
  constexpr int64_t cols_per_thread = 16, max_row_threads = 32768;
  const int64_t row_threads = std::clamp<int64_t>(cols / cols_per_thread, 1, max_row_threads);
  Tensor mask = at::empty({rows}, self.options().dtype(kInt));
  if (row_threads > 1) {
    mask.zero_();
  }

  const std::array<int64_t, 2> shape = {rows, cols};
  const std::string type_name = scalarToMetalTypeString(self);
  MPSStream* stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      id<MTLComputePipelineState> pso =
          lib.getPipelineStateForFunc(fmt::format("unique_mark_boundaries_{}", type_name));
      getMPSProfiler().beginProfileKernel(pso, "unique_mark_boundaries", false, stream);
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, sorted_values, mask, shape);
      mtl_dispatch2DJob(encoder, pso, row_threads, std::min(rows, max_threads));
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });

  // Step 3: inclusive scan of mask.
  const auto scan_dtype = rows <= std::numeric_limits<int32_t>::max() ? kInt : kLong;
  Tensor scan = at::cumsum(mask, /*dim=*/0, scan_dtype);
  const auto scan_suffix = std::to_string(c10::elementSize(scan_dtype) * 8);

  // Step 4: read num_unique.
  const int64_t num_unique = scan.select(0, rows - 1).item<int64_t>();
  TORCH_INTERNAL_ASSERT(num_unique >= 1);

  // Step 5: allocate outputs.
  Tensor unique_values = at::empty({num_unique, cols}, self.options());
  auto bound_pos = at::empty({return_counts ? num_unique : 0L}, self.options().dtype(kLong));
  auto counts = at::empty({return_counts ? num_unique : 0L}, self.options().dtype(kLong));
  auto inverse = at::empty({return_inverse ? rows : 0L}, self.options().dtype(kLong));
  const auto maybe = [](bool used, const Tensor& t) { return used ? std::optional<Tensor>(t) : std::nullopt; };
  const auto bound_pos_arg = maybe(return_counts, bound_pos);
  const auto sort_idx_arg = maybe(return_inverse && !consecutive, sort_idx);
  const auto inverse_arg = maybe(return_inverse, inverse);

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      {
        id<MTLComputePipelineState> pso =
            lib.getPipelineStateForFunc(fmt::format("unique_emit_{}_{}", type_name, scan_suffix));
        getMPSProfiler().beginProfileKernel(pso, "unique_emit", false, stream);
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, sorted_values, mask, scan, unique_values, bound_pos_arg, sort_idx_arg, inverse_arg, shape);
        mtl_dispatch2DJob(encoder, pso, row_threads, std::min(rows, max_threads));
        getMPSProfiler().endProfileKernel(pso, stream);
      }

      if (return_counts) {
        id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc("unique_counts");
        getMPSProfiler().beginProfileKernel(pso, "unique_counts", false, stream);
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, bound_pos, counts, static_cast<uint64_t>(num_unique), static_cast<uint64_t>(rows));
        mtl_dispatch1DJob(encoder, pso, std::min(num_unique, max_threads));
        getMPSProfiler().endProfileKernel(pso, stream);
      }
    }
  });

  return std::make_tuple(std::move(unique_values), std::move(inverse), std::move(counts));
}

static std::tuple<Tensor, Tensor, Tensor> _unique_impl_mps(const Tensor& self,
                                                           const bool return_inverse,
                                                           const bool return_counts,
                                                           const bool consecutive,
                                                           std::optional<int64_t> dimOpt) {
  TORCH_CHECK(isIntegralType(self.scalar_type(), /*includeBool=*/true) || mps::supportedFloatingType(self),
              "unique is not supported on MPS for dtype ",
              self.scalar_type());

  if (!dimOpt.has_value()) {
    if (self.numel() == 0) {
      Tensor output = at::empty({0}, self.options());
      Tensor inv = at::empty(return_inverse ? self.sizes() : IntArrayRef{0}, self.options().dtype(kLong));
      Tensor cnt = at::empty({0}, self.options().dtype(kLong));
      return std::make_tuple(std::move(output), std::move(inv), std::move(cnt));
    }
    const auto flat = self.contiguous().view({-1, 1});
    auto [values, inverse, counts] = _unique_rows_mps(flat, return_inverse, return_counts, consecutive);
    return {values.view(-1), return_inverse ? inverse.view(self.sizes()) : std::move(inverse), std::move(counts)};
  }

  const int64_t dim = dimOpt.value();
  auto sizes = self.sizes().vec();
  auto zero_dims = std::count(sizes.begin(), sizes.end(), (int64_t)0);
  if (self.size(dim) == 0) {
    TORCH_CHECK(zero_dims == 1, "Number of zero sized dimensions is more than one, so unique cannot be applied ");
    auto output = at::empty(sizes, self.options());
    auto inverse_indices = at::empty({0}, self.options().dtype(kLong));
    auto counts = at::empty({0}, self.options().dtype(kLong));
    return {std::move(output), std::move(inverse_indices), std::move(counts)};
  }
  TORCH_CHECK(zero_dims == 0, "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");

  auto transposed = self.moveaxis(dim, 0);
  auto orig_sizes = transposed.sizes().vec();
  auto input_flat = transposed.contiguous().view({orig_sizes[0], -1});
  auto [values, inverse, counts] = _unique_rows_mps(input_flat, return_inverse, return_counts, consecutive);
  orig_sizes[0] = values.size(0);
  return {values.view(orig_sizes).moveaxis(0, dim), std::move(inverse), std::move(counts)};
}

std::tuple<Tensor, Tensor, Tensor> unique_consecutive_mps(const Tensor& self,
                                                          const bool return_inverse,
                                                          const bool return_counts,
                                                          std::optional<int64_t> dim) {
  return _unique_impl_mps(self, return_inverse, return_counts, true, dim);
}

std::tuple<Tensor, Tensor, Tensor> unique_dim_consecutive_mps(const Tensor& self,
                                                              int64_t dim,
                                                              const bool return_inverse,
                                                              const bool return_counts) {
  return _unique_impl_mps(self, return_inverse, return_counts, true, dim);
}

std::tuple<Tensor, Tensor, Tensor> _unique2_mps(const Tensor& self,
                                                const bool sorted,
                                                const bool return_inverse,
                                                const bool return_counts) {
  return _unique_impl_mps(self, return_inverse, return_counts, false, std::nullopt);
}

std::tuple<Tensor, Tensor> _unique_mps(const Tensor& self, const bool sorted, const bool return_inverse) {
  auto [output, inverse_indices, _] = _unique_impl_mps(self, return_inverse, false, false, std::nullopt);
  return std::make_tuple(std::move(output), std::move(inverse_indices));
}

std::tuple<Tensor, Tensor, Tensor> unique_dim_mps(const Tensor& self,
                                                  int64_t dim,
                                                  const bool /*sorted*/,
                                                  const bool return_inverse,
                                                  const bool return_counts) {
  return _unique_impl_mps(self, return_inverse, return_counts, false, dim);
}

} // namespace at::native
