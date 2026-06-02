//  Hand-written Metal GEMM dispatcher for the MPS backend.
//
//  Ports the dispatch brain of metalBLAS (resolve_inputs, tile pickers, the
//  tensor-unit gate, and the autotuner) to host C++. Replaces the MPSGraph
//  matmul path for float/half/bfloat across mm/addmm/bmm/baddbmm/addbmm/addmv/
//  linear. int/complex stay on the existing naive-metal (do_metal_*) path.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/operations/GemmMetal.h>

#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/util/env.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native::mps {

namespace {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Gemm_metallib.h>
#endif

// (BM, BN, BK, WM, WN) for the simd kernel.
struct SimdTile {
  int BM, BN, BK, WM, WN;
};

// Describes how a kernel reads one logical matrix from possibly-strided memory.
struct Resolved {
  Tensor view; // original tensor or a contiguous copy
  int64_t ld; // leading dim (row stride, with the other dim unit-stride)
  bool trans; // true when stored column-major (the "other" orientation)
};

// metalBLAS _resolve_inputs, per matrix: rdim/cdim are the row/col dims of the
// logical 2-D matrix (the last two dims). Row-major contiguous keeps the view
// with ld = row stride; column-major is flagged transposed; anything else is
// made contiguous.
Resolved resolve_mat(const Tensor& m, int64_t rdim, int64_t cdim) {
  const int64_t R = m.size(rdim), C = m.size(cdim);
  const int64_t sr = m.stride(rdim), sc = m.stride(cdim);
  if (sc == 1 && sr >= C) {
    return {m, sr, false};
  }
  if (sr == 1 && sc >= R) {
    return {m, sc, true};
  }
  auto mc = m.contiguous();
  return {mc, C, false};
}

// metalBLAS _pick_simd_tile, trimmed to the instantiated tile set. Bounds are
// checked in-kernel, so an oversized tile is correct (just wasteful); the
// heuristic keeps enough tiles resident to fill the cores.
SimdTile pick_simd_tile(int64_t M, int64_t N, int64_t K) {
  const int64_t mx = std::max(M, N);
  const double ops = double(M) * double(N) * double(K);
  if (mx <= 64) {
    return {32, 32, 16, 1, 1};
  }
  if (ops > 256.0 * 1024 * 1024 && M >= 128 && N >= 128) {
    return {128, 128, 16, 4, 4};
  }
  return {64, 64, 16, 2, 2};
}

std::string simd_name(
    const std::string& dt,
    SimdTile t,
    bool trans_a,
    bool trans_b,
    at_gemm::GemmEpilogue epi,
    bool batched) {
  return fmt::format(
      "gemm_simd_{}_{}_{}_{}_{}_{}_ta{}_tb{}_{}_{}",
      dt,
      t.BM,
      t.BN,
      t.BK,
      t.WM,
      t.WN,
      trans_a ? 1 : 0,
      trans_b ? 1 : 0,
      epi == at_gemm::GemmEpilogue::AlphaBeta ? "ab" : "none",
      batched ? "b1" : "b0");
}

// (BM, BN, NSG) for the m5_tensor (matmul2d) kernel.
struct TensorTile {
  int BM, BN, NSG;
};

// metalBLAS _pick_m5_tensor_tile (M5 Pro sweeps). Every tile returned here must
// have a matching instantiation in Gemm.metal.
TensorTile pick_m5_tensor_tile(int64_t M, int64_t N, int64_t K, c10::ScalarType dt) {
  const int64_t mx = std::max(M, N);
  const bool is_lp = (dt != kFloat);
  const bool m_div_32 = (M % 32 == 0), m_div_64 = (M % 64 == 0);
  const bool n_div_64 = (N % 64 == 0), n_div_128 = (N % 128 == 0);
  if (M == 1 || N == 1) {
    return {16, 128, 4};
  }
  if (K <= 256 && M >= 1024 && N >= 1024 && m_div_32 && n_div_128) {
    return {32, 128, 4};
  }
  if (mx <= 256) {
    return {32, 32, 4};
  }
  if (M <= 48 && N >= 1024) {
    return (M <= 16) ? TensorTile{16, 64, 2} : TensorTile{32, 64, 2};
  }
  if (mx <= 1024) {
    const int64_t n64 = ((M + 63) / 64) * ((N + 63) / 64);
    if (n64 < 120 || (K <= mx && m_div_64 && n_div_64)) {
      return {32, 64, 2};
    }
  }
  if (K >= 2 * mx && mx >= 1792 && m_div_64 && n_div_128) {
    if (is_lp) {
      if (N < M && mx <= 2048) {
        return {64, 64, 2};
      }
      return {48, 128, 4};
    }
    return {64, 128, 4};
  }
  if (dt == kBFloat16 && mx >= 4096 && std::min(M, N) >= 1024 && !(m_div_64 && n_div_64)) {
    return {128, 128, 8};
  }
  if (is_lp) {
    return {64, 64, 2};
  }
  if (m_div_64 && n_div_128) {
    return {64, 128, 4};
  }
  return {64, 64, 2};
}

std::string m5t_name(
    const std::string& dt,
    TensorTile t,
    bool relaxed,
    at_gemm::GemmEpilogue epi,
    bool batched) {
  return fmt::format(
      "gemm_m5t_{}_{}_{}_{}_{}_{}_{}",
      dt,
      t.BM,
      t.BN,
      t.NSG,
      relaxed ? "relaxed" : "full",
      epi == at_gemm::GemmEpilogue::AlphaBeta ? "ab" : "none",
      batched ? "b1" : "b0");
}

} // namespace

bool gemm_supported_dtype(c10::ScalarType dt) {
  return dt == kFloat || dt == kHalf || dt == kBFloat16;
}

bool gemm_use_tensor_unit() {
  static const bool ok = []() {
    if (!is_macos_13_or_newer(MacOSVersion::MACOS_VER_26_4_PLUS)) {
      return false;
    }
    // MTLGPUFamilyApple10 (M5, the first tensor-unit family) only exists in the
    // macOS 26+ SDK; use the same fallback constant as Linear.mm.
    constexpr auto kApple10 = static_cast<MTLGPUFamily>(1010);
    return static_cast<bool>([MPSDevice::getInstance()->device() supportsFamily:kApple10]);
  }();
  return ok;
}

// The host packs dims into std::array<int32_t, N>; these guard against the
// kernel struct layout drifting from that packing.
static_assert(sizeof(at_gemm::GemmDimsStrided) == 13 * sizeof(int32_t));
static_assert(sizeof(at_gemm::GemmDimsPacked) == 9 * sizeof(int32_t));

void mps_gemm(
    const Tensor& A,
    const Tensor& B,
    const Tensor& out,
    const std::optional<Tensor>& self,
    const Scalar& alpha,
    const Scalar& beta,
    at_gemm::GemmEpilogue epi) {
  TORCH_INTERNAL_ASSERT(A.dim() == B.dim() && (A.dim() == 2 || A.dim() == 3));
  const bool batched = A.dim() == 3;
  const int64_t r = A.dim() - 2, c = A.dim() - 1;
  const int64_t batch = batched ? A.size(0) : 1;
  const int64_t M = A.size(r), K = A.size(c), N = B.size(c);
  TORCH_INTERNAL_ASSERT(B.size(r) == K);
  if (out.numel() == 0) {
    return;
  }

  const auto dt = out.scalar_type();
  const std::string dt_str = scalarToMetalTypeString(out);

  auto ra = resolve_mat(A, r, c);
  auto rb = resolve_mat(B, r, c);

  // Output must be row-major in its last dim for the C[row*ldc + col] store.
  Tensor target = (out.stride(c) == 1) ? out : at::empty(out.sizes(), out.options());
  const int64_t ldc = target.stride(r);

  // Epilogue addend, expanded to the output shape so broadcasts become stride-0.
  Tensor self_e;
  int64_t self_r = 0, self_c = 0, batch_self = 0;
  if (epi == at_gemm::GemmEpilogue::AlphaBeta) {
    TORCH_INTERNAL_ASSERT(self.has_value());
    self_e = self->expand_as(target);
    self_r = self_e.stride(r);
    self_c = self_e.stride(c);
    batch_self = batched ? self_e.stride(0) : 0;
  } else {
    self_e = A; // dummy binding for buffer(4); never dereferenced.
  }

  // Field order must match at_gemm::GemmDimsStrided (size asserted above).
  const std::array<int32_t, 13> dims = {
      static_cast<int32_t>(M),
      static_cast<int32_t>(N),
      static_cast<int32_t>(K),
      static_cast<int32_t>(ra.ld),
      static_cast<int32_t>(rb.ld),
      static_cast<int32_t>(ldc),
      static_cast<int32_t>(self_r),
      static_cast<int32_t>(self_c),
      /*swizzle_log=*/0,
      batched ? static_cast<int32_t>(ra.view.stride(0)) : 0,
      batched ? static_cast<int32_t>(rb.view.stride(0)) : 0,
      batched ? static_cast<int32_t>(target.stride(0)) : 0,
      static_cast<int32_t>(batch_self)};

  const std::array<float, 2> alpha_beta = {
      static_cast<float>(alpha.toDouble()), static_cast<float>(beta.toDouble())};

  // m5_tensor (tensor unit) handles packed, untransposed, contiguous-output
  // float GEMM on M5+; everything else falls to the portable simd kernel.
  const bool packed_in = !ra.trans && ra.ld == K && !rb.trans && rb.ld == N;
  const bool packed_out =
      target.stride(c) == 1 && ldc == N && (!batched || target.stride(0) == M * N);
  const bool use_m5t =
      gemm_use_tensor_unit() && packed_in && packed_out && M >= 2 && N >= 32 && K >= 64;

  auto stream = getCurrentMPSStream();

  if (use_m5t) {
    // fp32 defaults to TF32-relaxed tensor-unit math (the chosen MPS speed
    // default); set PYTORCH_MPS_PREFER_FP32_PRECISE for exact fp32. bf16/fp16
    // already accumulate in fp32, so they are inherently "relaxed".
    static const bool precise_fp32 = c10::utils::has_env("PYTORCH_MPS_PREFER_FP32_PRECISE");
    const bool relaxed = (dt != kFloat) || !precise_fp32;
    const TensorTile t = pick_m5_tensor_tile(M, N, K, dt);
    // Field order must match at_gemm::GemmDimsPacked (size asserted above).
    const std::array<int32_t, 9> pdims = {
        static_cast<int32_t>(M),
        static_cast<int32_t>(N),
        static_cast<int32_t>(K),
        static_cast<int32_t>(self_r),
        static_cast<int32_t>(self_c),
        batched ? static_cast<int32_t>(ra.view.stride(0)) : 0,
        batched ? static_cast<int32_t>(rb.view.stride(0)) : 0,
        batched ? static_cast<int32_t>(target.stride(0)) : 0,
        static_cast<int32_t>(batch_self)};
    const std::string fname = m5t_name(dt_str, t, relaxed, epi, batched);
    auto pso = lib.getPipelineStateForFunc(fname);
    const int64_t tiles_m = (M + t.BM - 1) / t.BM;
    const int64_t tiles_n = (N + t.BN - 1) / t.BN;
    const NSUInteger tg = static_cast<NSUInteger>(t.NSG * 32);
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        getMPSProfiler().beginProfileKernel(pso, "gemm_m5_tensor", {ra.view, rb.view});
        auto enc = stream->commandEncoder();
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, ra.view, rb.view, target, pdims, self_e, alpha_beta);
        [enc dispatchThreadgroups:MTLSizeMake(tiles_n, tiles_m, batch)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        getMPSProfiler().endProfileKernel(pso);
      }
    });
  } else {
    const SimdTile tile = pick_simd_tile(M, N, K);
    const std::string fname = simd_name(dt_str, tile, ra.trans, rb.trans, epi, batched);
    auto pso = lib.getPipelineStateForFunc(fname);
    const int64_t tiles_m = (M + tile.BM - 1) / tile.BM;
    const int64_t tiles_n = (N + tile.BN - 1) / tile.BN;
    const NSUInteger tg = static_cast<NSUInteger>(tile.WM * tile.WN * 32);
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        getMPSProfiler().beginProfileKernel(pso, "gemm_simd", {ra.view, rb.view});
        auto enc = stream->commandEncoder();
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, ra.view, rb.view, target, dims, self_e, alpha_beta);
        [enc dispatchThreadgroups:MTLSizeMake(tiles_n, tiles_m, batch)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        getMPSProfiler().endProfileKernel(pso);
      }
    });
  }

  if (!target.is_same(out)) {
    out.copy_(target);
  }
}

} // namespace at::native::mps
