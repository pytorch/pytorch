#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/ScanKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cummax_helper_native.h>
#include <ATen/ops/_cummin_helper_native.h>
#include <ATen/ops/_logcumsumexp_native.h>
#endif
#include <fmt/format.h>

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ScanKernel_metallib.h>
#endif

// Utility function to get 2D grid dimensions for dispatch
static std::pair<uint32_t, uint32_t> get_2d_grid_dims(const IntArrayRef& shape, const int64_t dim) {
  size_t grid_x = 1;
  size_t grid_y = 1;

  for (const auto i : c10::irange(dim)) {
    if (grid_x * shape[i] < UINT32_MAX) {
      grid_x *= shape[i];
    } else {
      grid_y *= shape[i];
    }
  }

  TORCH_CHECK(grid_y <= UINT32_MAX && grid_x <= UINT32_MAX, "Unable to safely factor shape for grid dimensions.");

  if (grid_y > grid_x) {
    std::swap(grid_x, grid_y);
  }

  return {static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y)};
}

// cumsum/cumprod over a dtype handled by the specialized Metal scan kernels.
static bool is_custom_scan_case(const std::string& op_name, ScalarType scalar_type) {
  if (op_name != "cumsum" && op_name != "cumprod") {
    return false;
  }
  switch (scalar_type) {
    case ScalarType::Float:
    case ScalarType::Half:
    case ScalarType::BFloat16:
    case ScalarType::Int:
    case ScalarType::Long:
      return true;
    default:
      return false;
  }
}

static ScalarType scan_accum_scalar_type(ScalarType st) {
  return at::isReducedFloatingType(st) ? ScalarType::Float : st;
}

// Few scans over a long axis: one-threadgroup-per-scan serializes it, so split
// each scan across threadgroups (multi-block).
static bool use_multiblock_scan(const std::string& op_name,
                                ScalarType scalar_type,
                                int64_t axis_size,
                                int64_t n_scans) {
  if (!is_custom_scan_case(op_name, scalar_type)) {
    return false;
  }
  constexpr int64_t kMinAxisSize = 1 << 16; // 65536
  constexpr int64_t kMaxScans = 1024;
  return axis_size >= kMinAxisSize && n_scans <= kMaxScans;
}

// Tiny innermost axis with very many scans: pack many rows per threadgroup
// (one-threadgroup-per-scan would launch far too many).
static bool use_tiny_scan(const std::string& op_name,
                          ScalarType scalar_type,
                          int64_t axis_size,
                          int64_t n_scans,
                          bool is_innermost) {
  if (!is_innermost || !is_custom_scan_case(op_name, scalar_type)) {
    return false;
  }
  constexpr int64_t kTinyAxis = 32;
  constexpr int64_t kMinScans = 4096;
  return axis_size >= 1 && axis_size <= kTinyAxis && n_scans >= kMinScans;
}

static void scan_tiny_innermost_mps_impl(const Tensor& input, const Tensor& output, const std::string& op_name) {
  const int64_t axis_size = input.size(-1);
  const int64_t n_scans = input.numel() / axis_size;
  constexpr int64_t kTinyTile = 2048; // must match TILE in scan_tiny_innermost
  const int64_t rows_per_tg = kTinyTile / axis_size;
  const int64_t num_tg = (n_scans + rows_per_tg - 1) / rows_per_tg;
  constexpr int64_t tg = 256;

  const auto type_str = scalarToMetalTypeString(input);
  const auto kernel_name = fmt::format("{}_tiny_innermost_{}", op_name, type_str);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(kernel_name);
      getMPSProfiler().beginProfileKernel(pso, op_name, {input, output});
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc,
                  input,
                  output,
                  static_cast<uint32_t>(axis_size),
                  static_cast<uint32_t>(n_scans),
                  static_cast<uint32_t>(rows_per_tg));
      [enc dispatchThreads:MTLSizeMake(tg * num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

// Three-pass parallel scan over a [n_scans, axis_size] contiguous tensor whose
// scan runs along the last (contiguous) dimension. See ScanKernel.metal.
static void scan_multiblock_mps_impl(const Tensor& input, const Tensor& output, const std::string& op_name) {
  const int64_t axis_size = input.size(-1);
  const int64_t n_scans = input.numel() / axis_size;

  const int n_reads = (input.element_size() <= 4) ? 4 : 2;
  constexpr int64_t tg = 256;
  const int64_t elems_per_iter = tg * n_reads;
  // Block count ~ sqrt(axis)/4 (empirical sweet spot over 65536..16M): enough
  // parallelism for long scans without over-subdividing short ones. Clamped below.
  const int64_t max_blocks_by_size = std::max<int64_t>(1, axis_size / elems_per_iter);
  int64_t num_blocks = static_cast<int64_t>(std::sqrt(static_cast<double>(axis_size))) / 4;
  num_blocks = std::max<int64_t>(num_blocks, 2);
  num_blocks = std::min<int64_t>(num_blocks, std::min<int64_t>(max_blocks_by_size, 4096));
  int64_t block_size = (axis_size + num_blocks - 1) / num_blocks;
  num_blocks = (axis_size + block_size - 1) / block_size;

  const auto acc_st = scan_accum_scalar_type(input.scalar_type());
  auto block_sums = at::empty({n_scans, num_blocks}, input.options().dtype(acc_st));

  const auto type_str = scalarToMetalTypeString(input);
  const auto reduce_name = fmt::format("{}_block_reduce_{}", op_name, type_str);
  const auto carry_name = fmt::format("{}_block_carry_{}", op_name, type_str);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();

      auto reducePSO = lib.getPipelineStateForFunc(reduce_name);
      getMPSProfiler().beginProfileKernel(reducePSO, op_name, {input});
      [enc setComputePipelineState:reducePSO];
      mtl_setArgs(enc, input, block_sums, axis_size, block_size, num_blocks);
      [enc dispatchThreads:MTLSizeMake(tg * num_blocks, n_scans, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      getMPSProfiler().endProfileKernel(reducePSO);

      // The carry pass computes each block's exclusive prefix inline from the
      // (cached) block_sums, so no separate block-sum scan pass is needed.
      auto carryPSO = lib.getPipelineStateForFunc(carry_name);
      getMPSProfiler().beginProfileKernel(carryPSO, op_name, {input, output});
      [enc setComputePipelineState:carryPSO];
      mtl_setArgs(enc, input, output, block_sums, axis_size, block_size, num_blocks);
      [enc dispatchThreads:MTLSizeMake(tg * num_blocks, n_scans, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      getMPSProfiler().endProfileKernel(carryPSO);
    }
  });
}

// Single-pass decoupled look-back over a [n_scans, axis_size] contiguous tensor
// (innermost scan), 1 read + 1 write. Float-accumulate dtypes only.
static void scan_decoupled_mps_impl(const Tensor& input, const Tensor& output, const std::string& op_name) {
  const int64_t axis_size = input.size(-1);
  const int64_t n_scans = input.numel() / axis_size;

  constexpr int64_t tg = 256;
  constexpr int64_t n_reads = 16; // must match REGISTER_DECOUPLED_SCAN_OP's NREADS
  constexpr int64_t tile = tg * n_reads;
  const int64_t num_tiles = (axis_size + tile - 1) / tile;
  const int64_t total_tiles = n_scans * num_tiles;

  // counter starts at 0; aggregate/inclusive slots start at sentinel kScanEmpty
  // (-1) so look-back spins until a producer publishes.
  auto counter = at::zeros({1}, input.options().dtype(kInt));
  auto aggregates = at::empty({total_tiles}, input.options().dtype(kInt));
  auto inclusive = at::empty({total_tiles}, input.options().dtype(kInt));
  aggregates.fill_(-1); // 0xFFFFFFFF == kScanEmpty
  inclusive.fill_(-1);

  const auto type_str = scalarToMetalTypeString(input);
  const auto kernel_name = fmt::format("{}_contig_decoupled_{}", op_name, type_str);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(kernel_name);
      getMPSProfiler().beginProfileKernel(pso, op_name, {input, output});
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc,
                  input,
                  output,
                  counter,
                  aggregates,
                  inclusive,
                  static_cast<uint32_t>(axis_size),
                  static_cast<uint32_t>(num_tiles));
      [enc dispatchThreads:MTLSizeMake(tg * total_tiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

// Multi-block scan for an outer (non-innermost) axis without transposing;
// contiguous input/output, scan stride n_irows = input.stride(wrapped_dim).
static void scan_strided_multiblock_mps_impl(const Tensor& input,
                                             const Tensor& output,
                                             int64_t wrapped_dim,
                                             const std::string& op_name) {
  const int64_t axis_size = input.size(wrapped_dim);
  const int64_t n_irows = input.stride(wrapped_dim);
  const int64_t n_orows = input.numel() / (axis_size * n_irows);
  const int64_t n_scans = n_orows * n_irows;

  const int n_reads = (input.element_size() <= 4) ? 4 : 2;
  constexpr int64_t BM = 32;
  // Fit the tile width to the scan stride so threadgroups aren't wasted on
  // padding columns (registered widths: 8, 16, 32).
  const int64_t BN = n_irows <= 8 ? 8 : (n_irows <= 16 ? 16 : 32);
  const int64_t stride_blocks = (n_irows + BN - 1) / BN;

  constexpr int64_t target_tg = 4096;
  const int64_t outer_tg = n_orows * stride_blocks;
  int64_t num_blocks = std::max<int64_t>((target_tg + outer_tg - 1) / outer_tg, 2);
  num_blocks = std::min<int64_t>(num_blocks, std::max<int64_t>(1, axis_size / BM));
  int64_t block_size = ((axis_size + num_blocks - 1) / num_blocks + BM - 1) / BM * BM;
  num_blocks = (axis_size + block_size - 1) / block_size;

  const auto acc_st = scan_accum_scalar_type(input.scalar_type());
  auto block_sums = at::empty({n_scans, num_blocks}, input.options().dtype(acc_st));

  const int64_t tg = (BN / n_reads) * 32;
  const int64_t total_groups = n_orows * stride_blocks * num_blocks;
  int64_t grid_y = total_groups, grid_z = 1;
  constexpr int64_t kMaxDim = 0x7fffffff;
  while (grid_y > kMaxDim) {
    grid_z *= 2;
    grid_y = (total_groups + grid_z - 1) / grid_z;
  }

  const auto type_str = scalarToMetalTypeString(input);
  const auto reduce_name = fmt::format("{}_strided_block_reduce_{}_{}", op_name, BN, type_str);
  const auto sums_name = fmt::format("{}_scan_block_sums_{}", op_name, type_str);
  const auto carry_name = fmt::format("{}_strided_block_carry_{}_{}", op_name, BN, type_str);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();

      auto reducePSO = lib.getPipelineStateForFunc(reduce_name);
      getMPSProfiler().beginProfileKernel(reducePSO, op_name, {input});
      [enc setComputePipelineState:reducePSO];
      mtl_setArgs(enc, input, block_sums, axis_size, n_irows, stride_blocks, block_size, num_blocks, n_orows);
      [enc dispatchThreads:MTLSizeMake(tg, grid_y, grid_z) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      getMPSProfiler().endProfileKernel(reducePSO);

      auto sumsPSO = lib.getPipelineStateForFunc(sums_name);
      getMPSProfiler().beginProfileKernel(sumsPSO, op_name, {block_sums});
      [enc setComputePipelineState:sumsPSO];
      mtl_setArgs(enc, block_sums, num_blocks);
      [enc dispatchThreads:MTLSizeMake(256, n_scans, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      getMPSProfiler().endProfileKernel(sumsPSO);

      auto carryPSO = lib.getPipelineStateForFunc(carry_name);
      getMPSProfiler().beginProfileKernel(carryPSO, op_name, {input, output});
      [enc setComputePipelineState:carryPSO];
      mtl_setArgs(enc, input, output, block_sums, axis_size, n_irows, stride_blocks, block_size, num_blocks, n_orows);
      [enc dispatchThreads:MTLSizeMake(tg, grid_y, grid_z) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      getMPSProfiler().endProfileKernel(carryPSO);
    }
  });
}

// Per-column general-stride decoupled look-back: one simdgroup per column-tile,
// constant N_READS regs, handles any inner stride. See ScanKernel.metal.
static void scan_strided_col_mps_impl(const Tensor& input,
                                      const Tensor& output,
                                      int64_t wrapped_dim,
                                      const std::string& op_name) {
  const int64_t axis_size = input.size(wrapped_dim);
  const int64_t n_irows = input.stride(wrapped_dim);
  const int64_t n_orows = input.numel() / (axis_size * n_irows);

  constexpr int64_t n_reads = 16; // must match REGISTER_STRIDED_COL_SCAN_OP's NREADS
  constexpr int64_t tile_rows = n_reads * 32;
  const int64_t num_tiles = (axis_size + tile_rows - 1) / tile_rows;
  const int64_t n_cols_total = n_orows * n_irows;
  const int64_t total_tiles = n_cols_total * num_tiles;

  constexpr int64_t simdgroups_per_tg = 8;
  const int64_t tg = simdgroups_per_tg * 32;
  const int64_t num_tg = (total_tiles + simdgroups_per_tg - 1) / simdgroups_per_tg;

  auto counter = at::zeros({1}, input.options().dtype(kInt));
  auto aggregates = at::empty({total_tiles}, input.options().dtype(kInt));
  auto inclusive = at::empty({total_tiles}, input.options().dtype(kInt));
  aggregates.fill_(-1); // 0xFFFFFFFF == kScanEmpty
  inclusive.fill_(-1);

  const auto type_str = scalarToMetalTypeString(input);
  const auto kernel_name = fmt::format("{}_strided_col_decoupled_{}", op_name, type_str);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(kernel_name);
      getMPSProfiler().beginProfileKernel(pso, op_name, {input, output});
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc,
                  input,
                  output,
                  counter,
                  aggregates,
                  inclusive,
                  static_cast<uint32_t>(axis_size),
                  static_cast<uint32_t>(n_irows),
                  static_cast<uint32_t>(n_cols_total),
                  static_cast<uint32_t>(total_tiles));
      [enc dispatchThreads:MTLSizeMake(tg * num_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

// Vectorized decoupled look-back: [axis, n_irows] as a contiguous scan over
// n_irows-vectors, coalesced + register-held. See ScanKernel.metal.
static void scan_vec_decoupled_mps_impl(const Tensor& input,
                                        const Tensor& output,
                                        int64_t wrapped_dim,
                                        const std::string& op_name) {
  const int64_t axis_size = input.size(wrapped_dim);
  const int64_t n_irows = input.stride(wrapped_dim);
  const int64_t n_orows = input.numel() / (axis_size * n_irows);

  // VEC == n_irows is compile-time; N_READS*n_irows ~ 32 registers per thread.
  const int64_t n_reads = std::max<int64_t>(1, 32 / n_irows);
  constexpr int64_t tg = 256;
  const int64_t tile = tg * n_reads; // super-elements per tile
  const int64_t num_tiles = (axis_size + tile - 1) / tile;
  const int64_t total_tiles = n_orows * num_tiles;

  auto counter = at::zeros({1}, input.options().dtype(kInt));
  auto aggregates = at::empty({total_tiles * n_irows}, input.options().dtype(kInt));
  auto inclusive = at::empty({total_tiles * n_irows}, input.options().dtype(kInt));
  aggregates.fill_(-1); // 0xFFFFFFFF == kScanEmpty
  inclusive.fill_(-1);

  const auto type_str = scalarToMetalTypeString(input);
  const auto kernel_name = fmt::format("{}_vec_decoupled_{}_{}", op_name, n_irows, type_str);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(kernel_name);
      getMPSProfiler().beginProfileKernel(pso, op_name, {input, output});
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc,
                  input,
                  output,
                  counter,
                  aggregates,
                  inclusive,
                  static_cast<uint32_t>(axis_size),
                  static_cast<uint32_t>(num_tiles));
      [enc dispatchThreads:MTLSizeMake(tg * total_tiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

void scan_simple_mps_impl(const Tensor& self, const Tensor& output, int64_t dim, const std::string& op_name) {
  if (output.numel() == 0) {
    return;
  }

  const int64_t ndim = self.dim();
  const int64_t wrapped_dim = maybe_wrap_dim(dim, ndim);
  const int64_t axis_size = self.size(wrapped_dim);
  const int64_t n_scans = self.numel() / axis_size;
  const bool is_innermost = (wrapped_dim == ndim - 1);

  // Run a scan that writes a contiguous result shaped like `proto`, delivering it
  // into `output` (staging through a temporary when output isn't contiguous).
  auto emit_scan = [&](const Tensor& proto, auto&& run) {
    if (output.is_contiguous()) {
      auto dst = output.view(proto.sizes());
      run(dst);
    } else {
      auto dst = at::empty_like(proto);
      run(dst);
      output.copy_(dst.view(output.sizes()));
    }
  };

  if (use_tiny_scan(op_name, self.scalar_type(), axis_size, n_scans, is_innermost)) {
    Tensor input_tensor = self.contiguous();
    emit_scan(input_tensor, [&](const Tensor& dst) { scan_tiny_innermost_mps_impl(input_tensor, dst, op_name); });
    return;
  }

  if (use_multiblock_scan(op_name, self.scalar_type(), axis_size, n_scans)) {
    // Multi-block kernels scan the last (contiguous) dim. Float-accumulate uses
    // the single-pass look-back; int/long keep the 3-pass path.
    const bool float_accum = self.scalar_type() == ScalarType::Float || self.scalar_type() == ScalarType::Half ||
        self.scalar_type() == ScalarType::BFloat16;
    Tensor input_tensor = self.contiguous();
    auto run = float_accum ? scan_decoupled_mps_impl : scan_multiblock_mps_impl;
    const int64_t n_irows = input_tensor.stride(wrapped_dim);
    if (is_innermost) {
      emit_scan(input_tensor, [&](const Tensor& dst) { run(input_tensor, dst, op_name); });
    } else if (n_irows == 1) {
      // Trailing dims are all size 1: the scan axis is effectively innermost, so
      // reshape to [n_scans, axis] and use the contiguous scan.
      auto in2 = input_tensor.reshape({n_scans, axis_size});
      emit_scan(in2, [&](const Tensor& dst) { run(in2, dst, op_name); });
    } else if (float_accum && n_irows == 2) {
      // Float-accumulate strided scans: vectorized kernel at n_irows==2, per-column
      // kernel for wider strides; int/long fall to the 3-pass path below.
      emit_scan(input_tensor,
                [&](const Tensor& dst) { scan_vec_decoupled_mps_impl(input_tensor, dst, wrapped_dim, op_name); });
    } else if (float_accum && n_irows >= 3) {
      emit_scan(input_tensor,
                [&](const Tensor& dst) { scan_strided_col_mps_impl(input_tensor, dst, wrapped_dim, op_name); });
    } else {
      emit_scan(input_tensor,
                [&](const Tensor& dst) { scan_strided_multiblock_mps_impl(input_tensor, dst, wrapped_dim, op_name); });
    }
    return;
  }

  // Preprocess input tensor - ensure it's contiguous for Metal shaders
  Tensor input_tensor = self.contiguous();

  // Preprocess output tensor - ensure it's contiguous for Metal shaders
  Tensor output_tensor = output;
  bool output_needs_copy = !output.is_contiguous();

  if (output_needs_copy) {
    // Create a temporary contiguous tensor with the same shape and type
    output_tensor = at::empty_like(output, output.options().memory_format(c10::MemoryFormat::Contiguous));
  }

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      // Build kernel name based on scan dimension position
      const auto type_str = scalarToMetalTypeString(input_tensor);
      const auto kernel_name = fmt::format("{}_{}_{}", op_name, is_innermost ? "innermost" : "outer", type_str);

      id<MTLComputePipelineState> scanPSO = lib.getPipelineStateForFunc(kernel_name);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(scanPSO, op_name, [&]() {
        std::vector<Tensor> all_tensors = {input_tensor, output_tensor};
        return all_tensors;
      }());

      [computeEncoder setComputePipelineState:scanPSO];

      // Set input and output buffers (both guaranteed contiguous)
      mtl_setArgs(computeEncoder, input_tensor, output_tensor);

      if (is_innermost) {
        // Contiguous scan dispatch (scanning innermost dimension)
        mtl_setArgs<2>(computeEncoder, axis_size);

        int n_reads = (input_tensor.element_size() <= 4) ? 4 : 2;
        constexpr int simd_size = 32;
        int elements_per_simd = n_reads * simd_size;
        int thread_group_size = static_cast<int>(scanPSO.maxTotalThreadsPerThreadgroup);

        if (axis_size <= n_reads * 1024) {
          thread_group_size = ((axis_size + elements_per_simd - 1) / elements_per_simd) * simd_size;
        } else if (axis_size <= n_reads * 2048) {
          thread_group_size = ((axis_size / 2 + elements_per_simd - 1) / elements_per_simd) * simd_size;
        }
        thread_group_size = std::min(thread_group_size, static_cast<int>(scanPSO.maxTotalThreadsPerThreadgroup));

        auto tmp_grid_dims = get_2d_grid_dims(input_tensor.sizes(), wrapped_dim);

        [computeEncoder dispatchThreads:MTLSizeMake(thread_group_size, tmp_grid_dims.first, tmp_grid_dims.second)
                  threadsPerThreadgroup:MTLSizeMake(thread_group_size, 1, 1)];
      } else {
        // Strided scan dispatch (scanning non-innermost dimension)
        size_t stride = input_tensor.strides()[wrapped_dim];
        constexpr int bn = 32;
        size_t stride_blocks = (stride + bn - 1) / bn;

        mtl_setArgs<2>(computeEncoder, axis_size, stride, stride_blocks);

        int n_reads = (input_tensor.element_size() <= 4) ? 4 : 2;
        int n_simdgroups = bn / n_reads;
        constexpr int simd_size = 32;
        int thread_group_size = n_simdgroups * simd_size;

        auto tmp_grid_dims = get_2d_grid_dims(input_tensor.sizes(), wrapped_dim);
        if (tmp_grid_dims.first * stride_blocks <= UINT_MAX) {
          tmp_grid_dims.first *= stride_blocks;
        } else {
          tmp_grid_dims.second *= stride_blocks;
        }

        [computeEncoder dispatchThreads:MTLSizeMake(thread_group_size, tmp_grid_dims.first, tmp_grid_dims.second)
                  threadsPerThreadgroup:MTLSizeMake(thread_group_size, 1, 1)];
      }

      getMPSProfiler().endProfileKernel(scanPSO);
    }
  });

  // Post-process: copy result back to original output tensor if needed
  if (output_needs_copy) {
    output.copy_(output_tensor);
  }
}

// Specialized implementation for cummin/cummax that returns both values and indices
static void scan_with_indices_mps_impl(const Tensor& self,
                                       const Tensor& values_output,
                                       const Tensor& indices_output,
                                       int64_t dim,
                                       const std::string& op_name) {
  if (values_output.numel() == 0) {
    return;
  }

  const int64_t ndim = self.dim();
  const int64_t wrapped_dim = maybe_wrap_dim(dim, ndim);
  const int64_t axis_size = self.size(wrapped_dim);

  // Preprocess input tensor - ensure it's contiguous for Metal shaders
  auto input_tensor = self.contiguous();

  // Preprocess output tensors - ensure they're contiguous for Metal shaders
  auto values_tensor = values_output.contiguous();
  auto indices_tensor = indices_output.contiguous();
  const bool values_needs_copy = !values_output.is_contiguous();
  const bool indices_needs_copy = !indices_output.is_contiguous();

  // Determine which kernel to use based on scan dimension position
  bool is_innermost_scan = (wrapped_dim == ndim - 1);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      // Build kernel name based on scan type
      const auto type_str = scalarToMetalTypeString(input_tensor);
      const auto kernel_name = fmt::format("{}_{}_{}", op_name, is_innermost_scan ? "innermost" : "outer", type_str);

      id<MTLComputePipelineState> scanPSO = lib.getPipelineStateForFunc(kernel_name);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(scanPSO, op_name, {input_tensor, values_tensor, indices_tensor});

      [computeEncoder setComputePipelineState:scanPSO];

      // Set input and output buffers (all guaranteed contiguous)
      mtl_setArgs(computeEncoder, input_tensor, values_tensor, indices_tensor);

      constexpr int simd_size = 32;

      if (is_innermost_scan) {
        // Contiguous scan dispatch (scanning innermost dimension)
        mtl_setArgs<3>(computeEncoder, axis_size);

        int n_reads = (input_tensor.element_size() <= 4) ? 4 : 2;

        int elements_per_simd = n_reads * simd_size;
        int thread_group_size = static_cast<int>(scanPSO.maxTotalThreadsPerThreadgroup);

        if (axis_size <= n_reads * 1024) {
          thread_group_size = ((axis_size + elements_per_simd - 1) / elements_per_simd) * simd_size;
        } else if (axis_size <= n_reads * 2048) {
          thread_group_size = ((axis_size / 2 + elements_per_simd - 1) / elements_per_simd) * simd_size;
        }
        thread_group_size = std::min(thread_group_size, static_cast<int>(scanPSO.maxTotalThreadsPerThreadgroup));

        auto tmp_grid_dims = get_2d_grid_dims(input_tensor.sizes(), wrapped_dim);

        [computeEncoder dispatchThreads:MTLSizeMake(thread_group_size, tmp_grid_dims.first, tmp_grid_dims.second)
                  threadsPerThreadgroup:MTLSizeMake(thread_group_size, 1, 1)];
      } else {
        // Strided scan dispatch (scanning non-innermost dimension)
        size_t stride = input_tensor.strides()[wrapped_dim];
        constexpr int bn = 32;
        size_t stride_blocks = (stride + bn - 1) / bn;

        mtl_setArgs<3>(computeEncoder, axis_size, stride, stride_blocks);

        int n_reads = (input_tensor.element_size() <= 4) ? 4 : 2;
        int n_simdgroups = bn / n_reads;
        int thread_group_size = n_simdgroups * simd_size;

        auto tmp_grid_dims = get_2d_grid_dims(input_tensor.sizes(), wrapped_dim);
        if (tmp_grid_dims.first * stride_blocks <= UINT_MAX) {
          tmp_grid_dims.first *= stride_blocks;
        } else {
          tmp_grid_dims.second *= stride_blocks;
        }

        [computeEncoder dispatchThreads:MTLSizeMake(thread_group_size, tmp_grid_dims.first, tmp_grid_dims.second)
                  threadsPerThreadgroup:MTLSizeMake(thread_group_size, 1, 1)];
      }

      getMPSProfiler().endProfileKernel(scanPSO);
    }
  });

  // Post-process: copy results back to original output tensors if needed
  if (values_needs_copy) {
    values_output.copy_(values_tensor);
  }
  if (indices_needs_copy) {
    indices_output.copy_(indices_tensor);
  }
}

} // namespace mps

void cummax_helper_mps(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  mps::scan_with_indices_mps_impl(self, values, indices, dim, "cummax");
}

void cummin_helper_mps(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  mps::scan_with_indices_mps_impl(self, values, indices, dim, "cummin");
}

Tensor& _logcumsumexp_out_mps(const Tensor& self, int64_t dim, Tensor& result) {
  const auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  result.resize_(self.sizes());
  if (self.dim() == 0) {
    result.fill_(self);
    return result;
  }
  if (self.numel() == 0) {
    result.zero_();
    return result;
  }

  mps::scan_simple_mps_impl(self, result, wrap_dim, "logcumsumexp");
  return result;
}

Tensor _logcumsumexp_mps(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  return _logcumsumexp_out_mps(self, dim, result);
}

} // namespace at::native
