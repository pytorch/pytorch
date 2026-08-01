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

// Kernel dtype tag: "{in}_{out}" when the scan widens (int32 -> int64), else the
// single type. The fused "{in}_{out}" kernel reads the narrow input directly.
static std::string scan_kernel_type_tag(const Tensor& input, const Tensor& output) {
  const auto in_str = scalarToMetalTypeString(input);
  if (input.scalar_type() == output.scalar_type()) {
    return in_str;
  }
  return in_str + "_" + scalarToMetalTypeString(output);
}

// Reads per thread, sized off the accumulator width (it sets register/smem use).
static int scan_n_reads(ScalarType accum_type) {
  return c10::elementSize(accum_type) <= 4 ? 4 : 2;
}

// Few scans over a long contiguous axis: one-threadgroup-per-scan serializes
// it, so split each scan across threadgroups (multi-block).
static bool use_contig_multiblock_scan(const std::string& op_name,
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

// scan_outer_dim is single-pass but unsplit, so it needs enough (orow x
// stride-block) groups to fill the GPU; only split to the multi-block kernels
// when that parallelism is starved (else the 3-pass split just adds overhead).
static bool use_strided_outer_scan(const std::string& op_name,
                                   ScalarType scalar_type,
                                   int64_t axis_size,
                                   int64_t n_orows,
                                   int64_t n_inner) {
  if (!is_custom_scan_case(op_name, scalar_type)) {
    return false;
  }
  constexpr int64_t kMinSplitAxis = 1024;
  if (axis_size < kMinSplitAxis) {
    return false;
  }
  const auto outer_groups = n_orows * ((n_inner + 31) / 32);
  constexpr int64_t kStarvedGroups = 64;
  return outer_groups <= kStarvedGroups;
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
  const auto axis_size = input.size(-1);
  const auto n_scans = input.numel() / axis_size;
  constexpr int64_t kTinyTile = 2048; // must match TILE in scan_tiny_innermost
  const auto rows_per_tg = kTinyTile / axis_size;
  const auto num_tg = (n_scans + rows_per_tg - 1) / rows_per_tg;
  constexpr int64_t tg = 256;

  const auto type_str = scan_kernel_type_tag(input, output);
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
  const auto axis_size = input.size(-1);
  const auto n_scans = input.numel() / axis_size;

  const auto acc_st = scan_accum_scalar_type(output.scalar_type());
  const auto n_reads = scan_n_reads(acc_st);
  constexpr int64_t tg = 256;
  const auto elems_per_iter = tg * n_reads;
  // Block count ~ sqrt(axis)/4 (empirical sweet spot over 65536..16M): enough
  // parallelism for long scans without over-subdividing short ones. Clamped below.
  const int64_t max_blocks_by_size = std::max<int64_t>(1, axis_size / elems_per_iter);
  auto num_blocks = static_cast<int64_t>(std::sqrt(static_cast<double>(axis_size))) / 4;
  num_blocks = std::max<int64_t>(num_blocks, 2);
  num_blocks = std::min<int64_t>(num_blocks, std::min<int64_t>(max_blocks_by_size, 4096));
  auto block_size = (axis_size + num_blocks - 1) / num_blocks;
  num_blocks = (axis_size + block_size - 1) / block_size;

  auto block_sums = at::empty({n_scans, num_blocks}, input.options().dtype(acc_st));

  const auto type_str = scan_kernel_type_tag(input, output);
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
  const auto axis_size = input.size(-1);
  const auto n_scans = input.numel() / axis_size;

  constexpr int64_t tg = 256;
  constexpr int64_t n_reads = 16; // must match REGISTER_DECOUPLED_SCAN_OP's NREADS
  constexpr int64_t tile = tg * n_reads;
  const auto num_tiles = (axis_size + tile - 1) / tile;
  const auto total_tiles = n_scans * num_tiles;

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

// NREADS per VEC, keeping NREADS*VEC near the contig path's register footprint.
// Must match the REGISTER_STRIDED_DECOUPLED_SCAN_OP_ALL_VEC (NREADS, VEC) pairs.
static int64_t strided_decoupled_n_reads(int64_t vec) {
  return vec <= 2 ? 8 : (vec <= 8 ? 4 : 2);
}

static bool strided_decoupled_supports_vec(int64_t vec) {
  return vec == 2 || vec == 4 || vec == 8 || vec == 16;
}

// Single-pass decoupled look-back for a narrow-stride outer scan over
// [n_orows, axis_size, VEC] contiguous: 1 dispatch, 1 read + 1 write, no global
// reduce->carry barrier. Float-accumulate dtypes only; gated to macOS 15+.
static void scan_strided_decoupled_mps_impl(const Tensor& input,
                                            const Tensor& output,
                                            int64_t wrapped_dim,
                                            const std::string& op_name) {
  const auto axis_size = input.size(wrapped_dim);
  const auto vec = input.stride(wrapped_dim);
  const auto n_orows = input.numel() / (axis_size * vec);

  // Coarse tiles minimize the per-tile fence/look-back count (swept vs MPSGraph).
  const int64_t tg = vec <= 2 ? 768 : 512;
  const auto n_reads = strided_decoupled_n_reads(vec);
  const auto tile = tg * n_reads;
  const auto num_tiles = (axis_size + tile - 1) / tile;
  const auto total_tiles = n_orows * num_tiles;

  auto counter = at::zeros({1}, input.options().dtype(kInt));
  auto status = at::zeros({total_tiles}, input.options().dtype(kInt));
  auto aggregates = at::empty({total_tiles * vec}, input.options().dtype(kInt));
  auto inclusive = at::empty({total_tiles * vec}, input.options().dtype(kInt));

  const auto type_str = scalarToMetalTypeString(input);
  const auto kernel_name = fmt::format("{}_strided_decoupled_{}_{}_{}", op_name, vec, n_reads, type_str);

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
                  status,
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
  const auto axis_size = input.size(wrapped_dim);
  const auto n_irows = input.stride(wrapped_dim);
  const auto n_orows = input.numel() / (axis_size * n_irows);
  const auto n_scans = n_orows * n_irows;

  const auto acc_st = scan_accum_scalar_type(output.scalar_type());
  const auto n_reads = scan_n_reads(acc_st);
  constexpr int64_t BM = 32;
  // Smallest registered tile width >= n_irows (no wasted padding columns). Int
  // has {8,16,32}; float also {12,24} for a tighter fit on strides in (8,24].
  const bool float_ext = output.scalar_type() == ScalarType::Float || at::isReducedFloatingType(output.scalar_type());
  const int64_t BN = float_ext
      ? (n_irows <= 8 ? 8 : (n_irows <= 12 ? 12 : (n_irows <= 16 ? 16 : (n_irows <= 24 ? 24 : 32))))
      : (n_irows <= 8 ? 8 : (n_irows <= 16 ? 16 : 32));
  const auto stride_blocks = (n_irows + BN - 1) / BN;

  constexpr int64_t target_tg = 4096;
  // Floor block size so tiny blocks don't over-subdivide a moderate axis (the
  // 3-pass overhead would dominate); long axes hit target_tg, not this floor.
  constexpr int64_t kMinBlock = 512;
  const auto outer_tg = n_orows * stride_blocks;
  auto num_blocks = std::max<int64_t>((target_tg + outer_tg - 1) / outer_tg, 2);
  num_blocks = std::min<int64_t>(num_blocks, std::max<int64_t>(1, axis_size / kMinBlock));
  auto block_size = ((axis_size + num_blocks - 1) / num_blocks + BM - 1) / BM * BM;
  num_blocks = (axis_size + block_size - 1) / block_size;

  auto block_sums = at::empty({n_scans, num_blocks}, input.options().dtype(acc_st));

  const auto tg = (BN / n_reads) * 32;
  const auto total_groups = n_orows * stride_blocks * num_blocks;
  int64_t grid_y = total_groups, grid_z = 1;
  constexpr int64_t kMaxDim = 0x7fffffff;
  while (grid_y > kMaxDim) {
    grid_z *= 2;
    grid_y = (total_groups + grid_z - 1) / grid_z;
  }

  const auto type_str = scan_kernel_type_tag(input, output);
  // block_sums is the accumulator type, so key its scan on the output dtype
  // (fused int32->int64 reuses the int64 sums kernel).
  const auto sums_type_str = scalarToMetalTypeString(output);
  const auto reduce_name = fmt::format("{}_strided_block_reduce_{}_{}", op_name, BN, type_str);
  const auto sums_name = fmt::format("{}_scan_block_sums_{}", op_name, sums_type_str);
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

// Int small-stride outer scan: 2-pass vectorized multi-block over
// [n_orows, axis, VEC]. Reads exactly VEC components; the strided kernel's
// min BN=8 tile would waste ~75% here.
static void scan_vec_multiblock_mps_impl(const Tensor& input,
                                         const Tensor& output,
                                         int64_t wrapped_dim,
                                         const std::string& op_name) {
  const auto axis_size = input.size(wrapped_dim);
  const auto vec = input.stride(wrapped_dim); // == n_irows == n_inner
  const auto n_orows = input.numel() / (axis_size * vec);

  const auto acc_st = scan_accum_scalar_type(output.scalar_type());
  const auto n_reads = scan_n_reads(acc_st);
  // Small tg for occupancy, coarse blocks to amortize the 2-pass overhead
  // (empirical sweet spot).
  constexpr int64_t tg = 64;
  constexpr int64_t target_groups = 512;
  const auto tile = tg * n_reads;
  // Enough (orow x block) groups to fill the GPU; block_size a whole tile count.
  auto num_blocks = std::max<int64_t>((target_groups + n_orows - 1) / n_orows, 2);
  num_blocks = std::min<int64_t>(num_blocks, std::max<int64_t>(1, axis_size / tile));
  num_blocks = std::max<int64_t>(num_blocks, 1);
  auto block_size = ((axis_size + num_blocks - 1) / num_blocks + tile - 1) / tile * tile;
  num_blocks = (axis_size + block_size - 1) / block_size;

  auto block_sums = at::empty({n_orows * num_blocks * vec}, input.options().dtype(acc_st));

  const auto type_str = scan_kernel_type_tag(input, output);
  const auto reduce_name = fmt::format("{}_vec_block_reduce_{}_{}", op_name, vec, type_str);
  const auto carry_name = fmt::format("{}_vec_block_carry_{}_{}", op_name, vec, type_str);

  const auto total_groups = n_orows * num_blocks;
  int64_t grid_y = total_groups, grid_z = 1;
  constexpr int64_t kMaxDim = 0x7fffffff;
  while (grid_y > kMaxDim) {
    grid_z *= 2;
    grid_y = (total_groups + grid_z - 1) / grid_z;
  }

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();

      auto reducePSO = lib.getPipelineStateForFunc(reduce_name);
      getMPSProfiler().beginProfileKernel(reducePSO, op_name, {input});
      [enc setComputePipelineState:reducePSO];
      mtl_setArgs(enc, input, block_sums, axis_size, block_size, num_blocks, n_orows);
      [enc dispatchThreads:MTLSizeMake(tg, grid_y, grid_z) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      getMPSProfiler().endProfileKernel(reducePSO);

      auto carryPSO = lib.getPipelineStateForFunc(carry_name);
      getMPSProfiler().beginProfileKernel(carryPSO, op_name, {input, output});
      [enc setComputePipelineState:carryPSO];
      mtl_setArgs(enc, input, output, block_sums, axis_size, block_size, num_blocks, n_orows);
      [enc dispatchThreads:MTLSizeMake(tg, grid_y, grid_z) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      getMPSProfiler().endProfileKernel(carryPSO);
    }
  });
}

// Fused transpose-scan: non-contiguous `input` whose innermost scan axis is
// physically outer. Storage is [axis_size, n_cols] contiguous (caller checked).
static void scan_innermost_transposed_mps_impl(const Tensor& input, const Tensor& output, const std::string& op_name) {
  const auto ndim = input.dim();
  const auto axis_size = input.size(ndim - 1);
  const auto n_cols = input.numel() / axis_size;
  constexpr int64_t BN = 32;
  const auto stride_blocks = (n_cols + BN - 1) / BN;

  const auto acc_st = scan_accum_scalar_type(output.scalar_type());
  const auto n_reads = scan_n_reads(acc_st);
  const auto tg = (BN / n_reads) * 32;

  int64_t grid_y = stride_blocks, grid_z = 1;
  constexpr int64_t kMaxDim = 0x7fffffff;
  while (grid_y > kMaxDim) {
    grid_z *= 2;
    grid_y = (stride_blocks + grid_z - 1) / grid_z;
  }

  const auto type_str = scan_kernel_type_tag(input, output);
  const auto kernel_name = fmt::format("{}_innermost_transposed_{}", op_name, type_str);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = mpsStream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(kernel_name);
      getMPSProfiler().beginProfileKernel(pso, op_name, {input, output});
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, input, output, axis_size, n_cols, stride_blocks);
      [enc dispatchThreads:MTLSizeMake(tg, grid_y, grid_z) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

// Dense non-contiguous input whose innermost dim is physically outer (a
// transpose): storage is contiguous when viewed axis-first.
static bool is_transposed_innermost_scan(const Tensor& self, const Tensor& output, int64_t wrapped_dim) {
  const auto ndim = self.dim();
  if (ndim < 2 || wrapped_dim != ndim - 1 || self.is_contiguous() || !output.is_contiguous()) {
    return false;
  }
  const auto n_cols = self.numel() / self.size(ndim - 1);
  constexpr int64_t kMinCols = 256; // enough column-blocks to fill the GPU
  if (n_cols < kMinCols) {
    return false;
  }
  std::vector<int64_t> perm(ndim);
  perm[0] = ndim - 1;
  for (int64_t i = 1; i < ndim; i++) {
    perm[i] = i - 1;
  }
  return self.permute(perm).is_contiguous();
}

void scan_simple_mps_impl(const Tensor& self, const Tensor& output, int64_t dim, const std::string& op_name) {
  if (output.numel() == 0) {
    return;
  }

  const auto ndim = self.dim();
  const auto wrapped_dim = maybe_wrap_dim(dim, ndim);
  const auto axis_size = self.size(wrapped_dim);
  const auto n_scans = self.numel() / axis_size;
  const bool is_innermost = (wrapped_dim == ndim - 1);

  // Run a scan that writes a contiguous result shaped like `proto`, delivering it
  // into `output` (staging through a temporary when output isn't contiguous).
  // The result carries `output`'s dtype (may differ from proto when widening).
  auto emit_scan = [&](const Tensor& proto, auto&& run) {
    if (output.is_contiguous()) {
      auto dst = output.view(proto.sizes());
      run(dst);
    } else {
      auto dst = at::empty(proto.sizes(), output.options());
      run(dst);
      output.copy_(dst.view(output.sizes()));
    }
  };

  // Transposed input scanning the innermost axis: fuse strided read + contiguous
  // write instead of a full .contiguous() copy.
  if (is_custom_scan_case(op_name, self.scalar_type()) && is_transposed_innermost_scan(self, output, wrapped_dim)) {
    scan_innermost_transposed_mps_impl(self, output, op_name);
    return;
  }

  if (use_tiny_scan(op_name, self.scalar_type(), axis_size, n_scans, is_innermost)) {
    Tensor input_tensor = self.contiguous();
    emit_scan(input_tensor, [&](const Tensor& dst) { scan_tiny_innermost_mps_impl(input_tensor, dst, op_name); });
    return;
  }

  // n_inner = columns between consecutive axis elements (contiguous layout);
  // n_inner == 1 means the axis is effectively innermost.
  int64_t n_inner = 1;
  for (int64_t i = wrapped_dim + 1; i < ndim; i++) {
    n_inner *= self.size(i);
  }
  const auto n_orows = n_scans / n_inner;

  // Float uses the single-pass decoupled look-back; int can't (no int64 in a
  // 32-bit atomic sentinel) and stays on the 3-pass kernels.
  const bool float_accum = self.scalar_type() == ScalarType::Float || self.scalar_type() == ScalarType::Half ||
      self.scalar_type() == ScalarType::BFloat16;

  // Float scans use the single-pass decoupled look-back on Apple9+ (M3 and
  // newer) running macOS 15+. The look-back hands a carry between threadgroups
  // via device-memory atomics, relying on cross-threadgroup ordering (a device
  // seq_cst fence) and forward progress that Apple8/M2 and older do not honor
  // reliably. macOS 14 (no device fence), older GPUs, and deterministic mode (timing-
  // dependent carry fold) use the deterministic multi-block kernels instead.
  const bool use_lookback = float_accum && is_macos_at_least(MacOSVersion::MACOS_15_0) &&
      is_apple_family_or_newer(AppleGPUFamily::APPLE_9_PLUS) && !globalContext().deterministicAlgorithms();

  if (n_inner == 1) {
    // Contiguous scan over the (effectively) innermost dim.
    if (use_contig_multiblock_scan(op_name, self.scalar_type(), axis_size, n_scans)) {
      Tensor input_tensor = self.contiguous();
      auto run = use_lookback ? scan_decoupled_mps_impl : scan_multiblock_mps_impl;
      auto in2 = is_innermost ? input_tensor : input_tensor.reshape({n_scans, axis_size});
      emit_scan(in2, [&](const Tensor& dst) { run(in2, dst, op_name); });
      return;
    }
  } else if (use_strided_outer_scan(op_name, self.scalar_type(), axis_size, n_orows, n_inner)) {
    Tensor input_tensor = self.contiguous();
    const auto n_irows = input_tensor.stride(wrapped_dim);
    // Narrow float strides {2,4,8,16}: single-pass look-back (1 dispatch, no
    // reduce->carry barrier) beats the multi-block on these latency-bound shapes.
    if (use_lookback && strided_decoupled_supports_vec(n_irows)) {
      emit_scan(input_tensor,
                [&](const Tensor& dst) { scan_strided_decoupled_mps_impl(input_tensor, dst, wrapped_dim, op_name); });
      return;
    }
    // Small strides read exactly VEC (vec multi-block); wider strides use the
    // 3-pass strided multi-block. Float covers {2..8}; int/long {2,3,4,8}.
    const bool use_vec =
        float_accum ? (n_irows >= 2 && n_irows <= 8) : ((n_irows >= 2 && n_irows <= 4) || n_irows == 8);
    if (use_vec) {
      // Small stride: 2-pass vectorized multi-block reads exactly VEC
      // components, avoiding the strided kernel's min BN=8 tile waste.
      emit_scan(input_tensor,
                [&](const Tensor& dst) { scan_vec_multiblock_mps_impl(input_tensor, dst, wrapped_dim, op_name); });
    } else {
      // Wide stride: the 3-pass strided multi-block kernel handles any stride.
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
      const auto type_str = scan_kernel_type_tag(input_tensor, output_tensor);
      const auto kernel_name = fmt::format("{}_{}_{}", op_name, is_innermost ? "innermost" : "outer", type_str);
      const auto n_reads = scan_n_reads(scan_accum_scalar_type(output_tensor.scalar_type()));

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

// Complex maps to "float2" in scalarToMetalTypeString, producing a non-existent kernel name.
void cummax_helper_mps(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  TORCH_CHECK(
      !c10::isComplexType(self.scalar_type()), "\"cummax\" not implemented for '", toString(self.scalar_type()), "'");
  mps::scan_with_indices_mps_impl(self, values, indices, dim, "cummax");
}

void cummin_helper_mps(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  TORCH_CHECK(
      !c10::isComplexType(self.scalar_type()), "\"cummin\" not implemented for '", toString(self.scalar_type()), "'");
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
