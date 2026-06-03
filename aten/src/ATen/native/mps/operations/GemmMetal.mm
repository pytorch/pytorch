//  Hand-written Metal GEMM dispatcher for the MPS backend.
//
//  Ports the dispatch brain of metalBLAS (resolve_inputs, tile pickers, the
//  tensor-unit gate, and the autotuner) to host C++. Replaces the MPSGraph
//  matmul path for float/half/bfloat across mm/addmm/bmm/baddbmm/addbmm/addmv/
//  linear. int/complex stay on the existing naive-metal (do_metal_*) path.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/operations/GemmMetal.h>

#include <ATen/mps/MPSProfiler.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/util/env.h>
#include <fmt/format.h>

#include <chrono>
#include <mutex>
#include <unordered_map>
#include <vector>

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
    bool trans_a,
    bool trans_b,
    bool relaxed,
    at_gemm::GemmEpilogue epi,
    bool batched) {
  return fmt::format(
      "gemm_m5t_{}_{}_{}_{}_ta{}_tb{}_{}_{}_{}",
      dt,
      t.BM,
      t.BN,
      t.NSG,
      trans_a ? 1 : 0,
      trans_b ? 1 : 0,
      relaxed ? "relaxed" : "full",
      epi == at_gemm::GemmEpilogue::AlphaBeta ? "ab" : "none",
      batched ? "b1" : "b0");
}

std::string gemv_t_name(const std::string& dt, int nw, int vec, at_gemm::GemmEpilogue epi) {
  return fmt::format(
      "gemv_t_{}_{}_{}_{}", dt, nw, vec, epi == at_gemm::GemmEpilogue::AlphaBeta ? "ab" : "none");
}

std::string gemv_nt_name(const std::string& dt, int nw, int vec, at_gemm::GemmEpilogue epi) {
  return fmt::format(
      "gemv_nt_{}_{}_{}_{}", dt, nw, vec, epi == at_gemm::GemmEpilogue::AlphaBeta ? "ab" : "none");
}

// c2 = float2 (complex64) / half2 (complex32).
std::string cgemv_t_name(const std::string& c2, int nw, at_gemm::GemmEpilogue epi) {
  return fmt::format(
      "cgemv_t_{}_{}_{}", c2, nw, epi == at_gemm::GemmEpilogue::AlphaBeta ? "ab" : "none");
}

std::string cgemv_nt_name(const std::string& c2, int nw, at_gemm::GemmEpilogue epi) {
  return fmt::format(
      "cgemv_nt_{}_{}_{}", c2, nw, epi == at_gemm::GemmEpilogue::AlphaBeta ? "ab" : "none");
}

// (NWARPS, VEC) for a GEMV launch. Integers keep VEC=1 (the validated path);
// float/half/bf16 use the metalBLAS _gemv_pick / _gemv_nt_pick heuristic - VEC
// scales with the output width for cache-line coverage, clamped to the matrix
// row-stride|offset alignment (a VEC-wide load needs both VEC-aligned). All
// returned (nw, vec) have a matching instantiation in Gemm.metal (MB_GEMV_*).
struct GemvCfg {
  int nw, vec;
};

// Largest power-of-2 <= vec that divides the row-stride|offset alignment (a
// VEC-wide load at (off + k*ld + n) is aligned for all k iff off and ld are).
int clamp_vec(int vec, int64_t align) {
  while (vec > 1 && (align % vec) != 0) {
    vec >>= 1;
  }
  return vec;
}

GemvCfg pick_gemv_t(
    c10::ScalarType dt,
    int64_t outlen,
    int64_t K,
    int64_t align,
    bool tensor_unit) {
  if (c10::isIntegralType(dt, /*includeBool=*/false)) {
    // _IGEMV_T_CFG: NW=8 always; VEC by element width (char/uchar 8, short 4,
    // int 2, long 1), clamped to alignment.
    const int64_t esz = c10::elementSize(dt);
    const int v = (esz == 1) ? 8 : (esz == 2) ? 4 : (esz == 4) ? 2 : 1;
    return {8, clamp_vec(v, align)};
  }
  if (dt == kFloat) {
    const int64_t ng = (outlen + 31) / 32;
    if (!tensor_unit) {
      return {ng >= 128 ? 16 : 32, 1};
    }
    return {(ng > 16 && ng <= 32) ? 16 : 32, 1};
  }
  const bool k_big = (K >= 2048);
  int nw, vec;
  if (!tensor_unit) {
    if (outlen > 12288) {
      vec = 8;
      nw = 8;
    } else if (outlen >= 2560) {
      vec = 2;
      nw = 32;
    } else {
      vec = 1;
      nw = 32;
    }
  } else if (outlen >= 4096 && (k_big || K >= 1024)) {
    vec = 8;
    nw = (K >= 8192) ? 16 : 8;
  } else if (outlen >= 2560 && (k_big || K >= 1024)) {
    vec = 4;
    nw = 8;
  } else if (k_big && outlen >= 1280) {
    vec = 4;
    nw = 16;
  } else if (outlen >= 1024) {
    vec = 2;
    nw = 16;
  } else if (outlen > 512) {
    vec = 2;
    nw = 32;
  } else {
    vec = 1;
    nw = 32;
  }
  // Clamp VEC to the row-stride|offset alignment.
  if (vec == 8 && (align & 7)) {
    if (!(align & 3)) {
      vec = 4;
      nw = 8;
    } else if (!(align & 1)) {
      vec = 2;
      nw = 32;
    } else {
      vec = 1;
      nw = 32;
    }
  } else if (vec == 4 && (align & 3)) {
    vec = !(align & 1) ? 2 : 1;
    nw = !(align & 1) ? 32 : 32;
  } else if (vec == 2 && (align & 1)) {
    vec = 1;
    nw = 32;
  }
  return {nw, vec};
}

GemvCfg pick_gemv_nt(c10::ScalarType dt, int64_t K, int64_t align, bool tensor_unit) {
  if (c10::isIntegralType(dt, /*includeBool=*/false)) {
    // _IGEMV_NT_CFG: char/uchar (VEC8,NW4), short (VEC4,NW4), int (VEC4,NW8),
    // long (VEC2,NW4); VEC clamped to alignment.
    const int64_t esz = c10::elementSize(dt);
    if (esz == 1) {
      return {4, clamp_vec(8, align)};
    }
    if (esz == 2) {
      return {4, clamp_vec(4, align)};
    }
    if (dt == kInt) {
      return {8, clamp_vec(4, align)};
    }
    return {4, clamp_vec(2, align)}; // long
  }
  if (dt == kFloat) {
    return {4, 1};
  }
  int vec;
  if (tensor_unit) {
    vec = (!(align & 3) && K >= 512) ? 4 : ((!(align & 1) && K >= 512) ? 2 : 1);
  } else {
    vec = (!(align & 3) && K >= 64) ? 4 : ((!(align & 1) && K >= 32) ? 2 : 1);
  }
  return {4, vec};
}

// (BM, BN, BK, TX, TY) for the register-tiled int_gemm kernel.
struct IntTile {
  int BM, BN, BK, TX, TY;
};

// metalBLAS _pick_int_tile (M5 Pro sweeps). Every tile must have a matching
// instantiation in Gemm.metal (MB_INT_ALL).
IntTile pick_int_tile(int64_t M, int64_t N, int64_t K, c10::ScalarType dt) {
  const int64_t nbytes = c10::elementSize(dt);
  const int64_t mx = std::max(M, N);
  if (mx <= 256) {
    return {64, 64, 16, 16, 16};
  }
  if (M <= 16 && N >= 1024) {
    return {16, 64, 16, 16, 16};
  }
  if (M <= 128 && N >= 1024) {
    return {32, 64, 16, 8, 16};
  }
  if (nbytes == 8) { // int64: shallow BK caps threadgroup memory
    return {64, 64, 8, 16, 16};
  }
  if (nbytes == 1 && M >= 512) { // int8/uint8 large: tall BM amortizes
    return {128, 64, 16, 16, 16};
  }
  return {64, 64, 16, 16, 16};
}

std::string int_name(
    const std::string& dt,
    IntTile t,
    bool trans_a,
    bool trans_b,
    at_gemm::GemmEpilogue epi,
    bool batched) {
  return fmt::format(
      "gemm_int_{}_{}_{}_{}_{}_{}_ta{}_tb{}_{}_{}",
      dt,
      t.BM,
      t.BN,
      t.BK,
      t.TX,
      t.TY,
      trans_a ? 1 : 0,
      trans_b ? 1 : 0,
      epi == at_gemm::GemmEpilogue::AlphaBeta ? "ab" : "none",
      batched ? "b1" : "b0");
}

// Deinterleave a complex buffer `src` into contiguous real planes re/im.
void launch_complex_split(const Tensor& src, const Tensor& re, const Tensor& im) {
  const int64_t n = src.numel();
  if (n == 0) {
    return;
  }
  auto pso = lib.getPipelineStateForFunc("complex_split_" + scalarToMetalTypeString(src));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "complex_split", {src});
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, src, re, im, static_cast<uint32_t>(n));
      mtl_dispatch1DJob(enc, pso, static_cast<uint32_t>(n));
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

// Fold the four real products into the interleaved complex result `dst`.
void launch_complex_combine(
    const Tensor& P,
    const Tensor& Q,
    const Tensor& S,
    const Tensor& T,
    const Tensor& dst) {
  const int64_t n = dst.numel();
  if (n == 0) {
    return;
  }
  auto pso = lib.getPipelineStateForFunc("complex_combine_" + scalarToMetalTypeString(dst));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "complex_combine", {P, Q});
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, P, Q, S, T, dst, static_cast<uint32_t>(n));
      mtl_dispatch1DJob(enc, pso, static_cast<uint32_t>(n));
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

// Native interleaved-complex rank-1 GEMV (M==1 / N==1). Reads the matrix once as
// C2 (vs the four-real-GEMM decomposition's repeated plane reads). a, b are
// already contiguous complex of the out dtype. Returns true when handled (the
// contiguous orientation cgemv reads); false to fall back to the decomposition.
bool launch_cgemv_complex(
    const Tensor& a,
    const Tensor& b,
    const Tensor& out,
    const std::optional<Tensor>& self,
    const Scalar& alpha,
    const Scalar& beta,
    at_gemm::GemmEpilogue epi) {
  const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
  if (((M == 1) == (N == 1))) {
    return false; // not rank-1 (or degenerate 1x1)
  }
  const bool m_is_one = (M == 1);
  const auto cdt = out.scalar_type();
  const std::string c2 = scalarToMetalTypeString(out);

  Tensor target = out.is_contiguous() ? out : at::empty(out.sizes(), out.options());

  Tensor self_e;
  int32_t self_r = 0, self_c = 0;
  if (epi == at_gemm::GemmEpilogue::AlphaBeta) {
    TORCH_INTERNAL_ASSERT(self.has_value());
    Tensor s = (self->scalar_type() != cdt) ? self->to(cdt) : self->contiguous();
    self_e = s.expand_as(target);
    self_r = static_cast<int32_t>(self_e.stride(0));
    self_c = static_cast<int32_t>(self_e.stride(1));
  } else {
    self_e = a; // dummy binding for buffer(4); never dereferenced
  }

  const int64_t outlen = m_is_one ? N : M;
  const Tensor& mat = m_is_one ? b : a; // M==1: matrix is B; N==1: matrix is A
  const Tensor& vec = m_is_one ? a : b;
  const int64_t ld = mat.stride(0); // ldb == N (cgemv_t) or lda == K (cgemv_nt)
  const std::array<int32_t, 6> gdims = {
      static_cast<int32_t>(outlen),
      static_cast<int32_t>(K),
      static_cast<int32_t>(ld),
      /*xs=*/1,
      m_is_one ? 0 : self_r, // cgemv_nt indexes self at (row, 0)
      m_is_one ? self_c : 0}; // cgemv_t indexes self at (0, n)

  const int nw = m_is_one ? 8 : 4;
  const std::string fname =
      m_is_one ? cgemv_t_name(c2, nw, epi) : cgemv_nt_name(c2, nw, epi);
  auto pso = lib.getPipelineStateForFunc(fname);
  const auto av = alpha.toComplexDouble();
  const auto bv = beta.toComplexDouble();
  const std::array<float, 4> ab = {
      static_cast<float>(av.real()),
      static_cast<float>(av.imag()),
      static_cast<float>(bv.real()),
      static_cast<float>(bv.imag())};
  const NSUInteger tg = static_cast<NSUInteger>(nw * 32);
  const int64_t ng = m_is_one ? ((outlen + 31) / 32) : ((outlen + nw - 1) / nw);

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "gemm_cgemv", {mat, vec});
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, mat, vec, target, gdims, self_e, ab);
      [enc dispatchThreadgroups:MTLSizeMake(ng, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });
  if (!target.is_same(out)) {
    out.copy_(target);
  }
  return true;
}

// ---------------------------------------------------------------------------
// Tile autotuner + process-global plan cache (port of metalBLAS dispatch.py).
// Ambiguous regimes (best tile flips with K / aspect) emit a candidate list that
// is probed once on private scratch and cached by (dtype, M, N, K, batched);
// candidate 0 is the heuristic, kept unless a challenger beats it by `margin`.
// Confident regimes return a single candidate (no probe). Packed-untransposed
// m5_tensor only - transposed / non-packed use the heuristic. Disable with
// PYTORCH_MPS_GEMM_AUTOTUNE=0.
// ---------------------------------------------------------------------------
constexpr double kAutotuneMargin = 0.03;
constexpr double kTallNarrowMargin = 0.01;

bool autotune_enabled() {
  static const bool on = []() {
    auto v = c10::utils::get_env("PYTORCH_MPS_GEMM_AUTOTUNE");
    return !(v.has_value() && v.value() == "0");
  }();
  return on;
}

std::vector<TensorTile> with_primary(
    TensorTile primary,
    std::vector<TensorTile> extra) {
  std::vector<TensorTile> c{primary};
  for (auto t : extra) {
    bool dup = false;
    for (auto& e : c) {
      if (e.BM == t.BM && e.BN == t.BN && e.NSG == t.NSG) {
        dup = true;
        break;
      }
    }
    if (!dup) {
      c.push_back(t);
    }
  }
  return c;
}

// metalBLAS _mpp_tensor_tile_candidates: (candidates, margin). Candidate 0 is the
// heuristic; a longer list marks an ambiguous regime. Every tile here must be an
// instantiated untransposed m5t tile (Gemm.metal MB_M5T_* + MB_M5T_UNTRANS).
std::pair<std::vector<TensorTile>, double> mpp_tensor_tile_candidates(
    int64_t M,
    int64_t N,
    int64_t K,
    c10::ScalarType dt) {
  TensorTile primary = pick_m5_tensor_tile(M, N, K, dt);
  if (dt == kFloat) {
    return {{primary}, kAutotuneMargin}; // fp32 wins everywhere - never probe
  }
  const int64_t mx = std::max(M, N), mn = std::min(M, N);
  if (M == 1 || N == 1) {
    return {{primary}, kAutotuneMargin};
  }
  if (K <= 256 && M >= 1024 && N >= 1024) {
    return {{primary}, kAutotuneMargin};
  }
  if (mx <= 256) {
    return {{primary}, kAutotuneMargin};
  }
  if (mn <= 256 && mx >= 1024) {
    std::vector<TensorTile> extra = {
        {128, 32, 2}, {256, 32, 4}, {32, 128, 2}, {32, 256, 4},
        {64, 64, 2}, {64, 32, 2}, {32, 64, 2}};
    if (mn <= 48) {
      extra.insert(extra.begin(), {{16, 64, 2}, {32, 64, 2}, {32, 128, 4}});
    }
    return {with_primary(primary, extra), kTallNarrowMargin};
  }
  if (mx <= 1024) {
    if (K >= 8 * mx) {
      return {with_primary(primary, {{128, 32, 4}, {128, 32, 2}, {192, 32, 2}, {32, 64, 2}, {64, 64, 2}}),
              kTallNarrowMargin};
    }
    return {with_primary(primary, {{16, 64, 2}, {32, 64, 2}, {64, 64, 2}}), kAutotuneMargin};
  }
  std::vector<TensorTile> extra = {{64, 64, 2}, {48, 128, 4}, {64, 128, 4}, {128, 64, 4}};
  if (mx >= 2048 && !(M % 64 == 0 && N % 64 == 0)) {
    extra.push_back({128, 128, 8});
  }
  return {with_primary(primary, extra), kAutotuneMargin};
}

// metalBLAS _pick_bmm_tile: the batch already fills the cores, so bmm prefers a
// bigger tile (more K reuse) + NSG=4 over the 2-D heuristic's tiny tiles.
TensorTile pick_bmm_tile(int64_t M, int64_t N, int64_t K, c10::ScalarType dt) {
  if (M == 1 || N == 1) {
    return pick_m5_tensor_tile(M, N, K, dt);
  }
  const int64_t mx = std::max(M, N), mn = std::min(M, N);
  if (K <= 128 && mx >= 512) {
    return {32, 128, 4};
  }
  if (mn <= 32) {
    return {32, 32, 4};
  }
  if (mx <= 256) {
    return {64, 64, 4};
  }
  return {64, 64, 2};
}

std::vector<TensorTile> bmm_candidates(int64_t M, int64_t N, int64_t K, c10::ScalarType dt) {
  static const TensorTile kBmmTiles[] = {
      {64, 64, 2}, {64, 64, 4}, {32, 64, 2}, {32, 128, 4},
      {64, 128, 4}, {128, 64, 4}, {128, 32, 4}, {128, 128, 8}};
  std::vector<TensorTile> extra;
  for (auto t : kBmmTiles) {
    if (t.BM <= 2 * M && t.BN <= 2 * N) {
      extra.push_back(t);
    }
  }
  return with_primary(pick_bmm_tile(M, N, K, dt), extra);
}

// A resolved launch plan: an m5_tensor tile, or a split-K / 1x1-conv candidate
// (deep-K / thin-N regimes the autotuner verifies against the m5_tensor result).
struct GemmPlan {
  enum Backend { kM5T, kSplitK, kConv } backend = kM5T;
  TensorTile tile{64, 64, 2}; // m5t tile, or split-K (BM, BN, NSG)
  int G = 0; // split-K chunks
  int BMW = 0, BNO = 0, convNSG = 0; // conv
};

// metalBLAS _is_splitk_regime / _is_conv_regime + spec families (low precision).
struct SplitKSpec {
  int BM, BN, NSG, G;
};
struct ConvSpec {
  int BMW, BNO, NSG;
};

bool is_splitk_regime(int64_t M, int64_t N, int64_t K, c10::ScalarType dt) {
  return dt != kFloat && K >= 2048 && std::min(M, N) >= 64 && M * N <= 1500000 &&
      (std::min(M, N) <= 256 || K >= 8 * std::max(M, N));
}
std::vector<SplitKSpec> splitk_specs(int64_t K) {
  std::vector<SplitKSpec> s;
  const int tiles[][3] = {{128, 32, 2}, {64, 64, 2}, {32, 64, 2}};
  for (auto& t : tiles) {
    for (int G : {2, 4}) {
      if (K % G == 0 && (K / G) % 16 == 0) {
        s.push_back({t[0], t[1], t[2], G});
      }
    }
  }
  return s;
}
// conv needs a COMPILE-TIME channel count (KCONST), so it is enabled only for the
// precompiled K set (the AOT analog of metalBLAS's JIT-per-K); other K -> m5_tensor.
bool conv_k_supported(int64_t K) {
  return K == 512 || K == 1024 || K == 2048 || K == 4096;
}
bool is_conv_regime(int64_t M, int64_t N, int64_t K, c10::ScalarType dt) {
  return dt != kFloat && N <= 64 && N % 32 == 0 && M >= 512 && conv_k_supported(K);
}
std::vector<ConvSpec> conv_specs(int64_t M, int64_t N) {
  std::vector<ConvSpec> s;
  for (int BMW : {64, 128}) { // the instantiated conv tile widths
    if (M % BMW == 0) {
      for (int NSG : {2, 4}) {
        s.push_back({BMW, static_cast<int>(N), NSG});
      }
    }
  }
  return s;
}

// Two-pass split-K into a packed (ldc == N) output; encode-only (no commit).
void launch_splitk(
    const std::string& dt_str,
    const Tensor& A,
    const Tensor& B,
    const Tensor& out,
    int64_t M,
    int64_t N,
    int64_t K,
    int BM,
    int BN,
    int NSG,
    int G) {
  const int planes = G; // every chunk (incl. 0) writes an fp32 plane
  const int kchunk = static_cast<int>(K / G);
  Tensor Cp = at::empty({planes, M, N}, out.options().dtype(kFloat));
  auto stream = getCurrentMPSStream();
  auto pso_g = lib.getPipelineStateForFunc(
      fmt::format("splitk_gemm_{}_{}_{}_{}", dt_str, BM, BN, NSG));
  auto pso_r = lib.getPipelineStateForFunc("splitk_reduce_" + dt_str);
  const int64_t tiles_m = (M + BM - 1) / BM;
  const int64_t tiles_n = (N + BN - 1) / BN;
  const std::array<int32_t, 4> sk = {
      static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K), kchunk};
  const std::array<int32_t, 2> rd = {static_cast<int32_t>(M * N), planes};
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso_g, "splitk_gemm", {A, B});
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso_g];
      mtl_setArgs(enc, A, B, Cp, sk);
      [enc dispatchThreadgroups:MTLSizeMake(tiles_n, tiles_m, G)
          threadsPerThreadgroup:MTLSizeMake(NSG * 32, 1, 1)];
      getMPSProfiler().endProfileKernel(pso_g);
    }
  });
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso_r, "splitk_reduce", {Cp});
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso_r];
      mtl_setArgs(enc, Cp, out, rd);
      mtl_dispatch1DJob(enc, pso_r, static_cast<uint32_t>(M * N));
      getMPSProfiler().endProfileKernel(pso_r);
    }
  });
}

// 1x1-conv GEMM into a packed (ldc == N) output; encode-only (no commit).
void launch_conv(
    const std::string& dt_str,
    const Tensor& A,
    const Tensor& B,
    const Tensor& out,
    int64_t M,
    int64_t N,
    int64_t K,
    int BMW,
    int BNO,
    int NSG) {
  auto stream = getCurrentMPSStream();
  auto pso = lib.getPipelineStateForFunc(
      fmt::format("conv1x1_gemm_{}_{}_{}_{}_{}", dt_str, BMW, BNO, NSG, K));
  const int64_t tiles_o = (N + BNO - 1) / BNO;
  const int64_t tiles_w = (M + BMW - 1) / BMW;
  const std::array<int32_t, 3> cd = {
      static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K)};
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "conv1x1_gemm", {A, B});
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, A, B, out, cd);
      [enc dispatchThreadgroups:MTLSizeMake(tiles_o, tiles_w, 1)
          threadsPerThreadgroup:MTLSizeMake(NSG * 32, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

struct PlanKey {
  c10::ScalarType dt;
  int64_t M, N, K;
  bool batched;
  bool operator==(const PlanKey& o) const {
    return dt == o.dt && M == o.M && N == o.N && K == o.K && batched == o.batched;
  }
};
struct PlanKeyHash {
  size_t operator()(const PlanKey& k) const {
    size_t h = std::hash<int>()(static_cast<int>(k.dt));
    auto mix = [&](int64_t v) { h ^= std::hash<int64_t>()(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
    mix(k.M);
    mix(k.N);
    mix(k.K);
    mix(k.batched ? 1 : 0);
    return h;
  }
};
std::unordered_map<PlanKey, GemmPlan, PlanKeyHash> g_tile_cache;
std::mutex g_tile_mutex;

// Time each candidate (best-of-reps min) on private scratch, returning the
// fastest plan. Candidate 0 (heuristic m5t) is kept unless beaten by > margin.
// split-K / conv candidates are verified against the m5t reference (max rel diff
// <= 2%) before joining the probe. Mirrors metalBLAS _autotune_mppt: warm every
// candidate first, then interleave timed reps.
GemmPlan autotune_plan(
    const std::string& dt_str,
    bool relaxed,
    bool batched,
    const Tensor& Av,
    const Tensor& Bv,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t batch,
    const std::vector<TensorTile>& cands,
    double margin,
    const std::vector<SplitKSpec>& sk_specs,
    const std::vector<ConvSpec>& conv_specs) {
  auto stream = getCurrentMPSStream();
  Tensor scratch = batched ? at::empty({batch, M, N}, Av.options())
                           : at::empty({M, N}, Av.options());
  // Packed (ta0 tb0) dims for the scratch (lda=K, ldb=N, ldc=N).
  const std::array<int32_t, 13> dims = {
      static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K),
      static_cast<int32_t>(K), static_cast<int32_t>(N), static_cast<int32_t>(N),
      0, 0, 0,
      batched ? static_cast<int32_t>(M * K) : 0,
      batched ? static_cast<int32_t>(K * N) : 0,
      batched ? static_cast<int32_t>(M * N) : 0,
      0};
  const std::array<float, 2> ab = {1.0f, 0.0f};

  auto enqueue_m5t = [&](TensorTile t, const Tensor& o) {
    const std::string fname =
        m5t_name(dt_str, t, false, false, relaxed, at_gemm::GemmEpilogue::None, batched);
    auto pso = lib.getPipelineStateForFunc(fname);
    const int64_t tiles_m = (M + t.BM - 1) / t.BM;
    const int64_t tiles_n = (N + t.BN - 1) / t.BN;
    const NSUInteger tg = static_cast<NSUInteger>(t.NSG * 32);
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto enc = stream->commandEncoder();
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, Av, Bv, o, dims, Av, ab);
        [enc dispatchThreadgroups:MTLSizeMake(tiles_n, tiles_m, batch)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      }
    });
  };

  struct Cand {
    GemmPlan plan;
    std::function<void(const Tensor&)> run;
  };
  std::vector<Cand> cand;
  for (auto t : cands) {
    cand.push_back({GemmPlan{GemmPlan::kM5T, t}, [&, t](const Tensor& o) { enqueue_m5t(t, o); }});
  }

  // split-K / conv candidates: include only when their result matches the m5t
  // reference EVERYWHERE. An allclose-style per-element gate (|o-ref| <=
  // rtol*|ref| + atol) - NOT a global-scale max-abs - because a chunked/reshaped
  // kernel that is wrong only at a few small-magnitude outputs would slip past a
  // global tolerance and then be selected, returning a subtly wrong result.
  if (!sk_specs.empty() || !conv_specs.empty()) {
    Tensor ref = at::empty({M, N}, Av.options());
    cand[0].run(ref);
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
    auto matches = [&](const Tensor& o) {
      // both kernels accumulate in fp32, so a correct candidate differs from the
      // m5t reference only by bf16/fp16 rounding (~1 ULP); 5% rtol + small atol.
      return o.sub(ref).abs().sub(ref.abs().mul(0.05)).max().item<double>() <= 0.05;
    };
    Tensor probe = at::empty({M, N}, Av.options());
    for (auto sp : sk_specs) {
      launch_splitk(dt_str, Av, Bv, probe, M, N, K, sp.BM, sp.BN, sp.NSG, sp.G);
      stream->synchronize(SyncType::COMMIT_AND_WAIT);
      if (matches(probe)) {
        GemmPlan p;
        p.backend = GemmPlan::kSplitK;
        p.tile = {sp.BM, sp.BN, sp.NSG};
        p.G = sp.G;
        cand.push_back({p, [&, sp](const Tensor& o) {
                          launch_splitk(dt_str, Av, Bv, o, M, N, K, sp.BM, sp.BN, sp.NSG, sp.G);
                        }});
      }
    }
    for (auto sp : conv_specs) {
      launch_conv(dt_str, Av, Bv, probe, M, N, K, sp.BMW, sp.BNO, sp.NSG);
      stream->synchronize(SyncType::COMMIT_AND_WAIT);
      if (matches(probe)) {
        GemmPlan p;
        p.backend = GemmPlan::kConv;
        p.BMW = sp.BMW;
        p.BNO = sp.BNO;
        p.convNSG = sp.NSG;
        cand.push_back({p, [&, sp](const Tensor& o) {
                          launch_conv(dt_str, Av, Bv, o, M, N, K, sp.BMW, sp.BNO, sp.NSG);
                        }});
      }
    }
  }

  // (warmup, iters, reps) scaled by FLOPs (metalBLAS _probe_params).
  const double flops = double(M) * double(N) * double(K) * double(batch);
  int warmup, iters, reps;
  if (flops <= 2e9) {
    warmup = 20;
    iters = 80;
    reps = 8;
  } else if (flops <= 5e10) {
    warmup = 3;
    iters = 3;
    reps = 3;
  } else {
    warmup = 2;
    iters = 3;
    reps = 3;
  }
  if (margin < kAutotuneMargin) {
    iters = std::max(iters * 2, 6);
    reps = std::max(reps, 5);
  }

  for (auto& c : cand) {
    for (int w = 0; w < warmup; ++w) {
      c.run(scratch);
    }
  }
  stream->synchronize(SyncType::COMMIT_AND_WAIT);

  std::vector<double> best(cand.size(), 1e30);
  for (int r = 0; r < reps; ++r) {
    for (size_t j = 0; j < cand.size(); ++j) {
      auto t0 = std::chrono::steady_clock::now();
      for (int it = 0; it < iters; ++it) {
        cand[j].run(scratch);
      }
      stream->synchronize(SyncType::COMMIT_AND_WAIT);
      const double dt =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters;
      best[j] = std::min(best[j], dt);
    }
  }
  size_t bi = 0;
  for (size_t j = 1; j < best.size(); ++j) {
    if (best[j] < best[bi]) {
      bi = j;
    }
  }
  if (bi != 0 && best[bi] < best[0] * (1.0 - margin)) {
    return cand[bi].plan;
  }
  return cand[0].plan;
}

// Resolve the launch plan: heuristic m5t for transposed / non-packed / autotune-off,
// else the cached autotuned winner (probing once on first sight of the shape).
GemmPlan choose_gemm_plan(
    c10::ScalarType dt,
    const std::string& dt_str,
    bool relaxed,
    bool batched,
    bool packed_untrans,
    const Tensor& Av,
    const Tensor& Bv,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t batch) {
  if (!autotune_enabled() || !packed_untrans) {
    GemmPlan p;
    p.tile = batched ? pick_bmm_tile(M, N, K, dt) : pick_m5_tensor_tile(M, N, K, dt);
    return p;
  }
  const PlanKey key{dt, M, N, K, batched};
  {
    std::lock_guard<std::mutex> g(g_tile_mutex);
    auto it = g_tile_cache.find(key);
    if (it != g_tile_cache.end()) {
      return it->second;
    }
  }
  std::vector<TensorTile> cands;
  double margin = kAutotuneMargin;
  if (batched) {
    cands = bmm_candidates(M, N, K, dt);
  } else {
    auto pr = mpp_tensor_tile_candidates(M, N, K, dt);
    cands = pr.first;
    margin = pr.second;
  }
  std::vector<SplitKSpec> sk;
  std::vector<ConvSpec> cv;
  if (!batched) {
    if (is_splitk_regime(M, N, K, dt)) {
      sk = splitk_specs(K);
    }
    if (is_conv_regime(M, N, K, dt)) {
      cv = conv_specs(M, N);
    }
  }
  GemmPlan plan;
  if (cands.size() > 1 || !sk.empty() || !cv.empty()) {
    plan = autotune_plan(dt_str, relaxed, batched, Av, Bv, M, N, K, batch, cands, margin, sk, cv);
  } else {
    plan.backend = GemmPlan::kM5T;
    plan.tile = cands[0];
  }
  static const bool debug = c10::utils::has_env("PYTORCH_MPS_GEMM_DEBUG");
  if (debug) {
    const char* bk = plan.backend == GemmPlan::kSplitK ? "splitk"
        : plan.backend == GemmPlan::kConv             ? "conv"
                                                      : "m5t";
    TORCH_WARN(
        "[mps_gemm] ",
        batched ? "bmm " : "",
        bk,
        " M=", M, " N=", N, " K=", K,
        " tile=", plan.tile.BM, "x", plan.tile.BN, "x", plan.tile.NSG,
        plan.backend == GemmPlan::kSplitK ? (" G=" + std::to_string(plan.G)) : "",
        plan.backend == GemmPlan::kConv
            ? (" BMW=" + std::to_string(plan.BMW) + " BNO=" + std::to_string(plan.BNO))
            : "");
  }
  {
    std::lock_guard<std::mutex> g(g_tile_mutex);
    g_tile_cache.emplace(key, plan); // first writer wins
  }
  return plan;
}

} // namespace

bool gemm_supported_dtype(c10::ScalarType dt) {
  return dt == kFloat || dt == kHalf || dt == kBFloat16 ||
      c10::isIntegralType(dt, /*includeBool=*/false);
}

bool gemm_use_mpp() {
  // matmul2d (MetalPerformancePrimitives) is gated only on the OS deployment
  // target - the SDK exposes it at macOS 26.2+ with no GPU-family requirement, and
  // it lowers to the NAX matrix unit where present and to simdgroup execution
  // otherwise. macOS 26.4 also guarantees kernels_40.metallib (the metal4.0 build
  // holding these kernels) is loaded; below it only the metal3.1 simd/gemv kernels
  // exist, so the simd path is taken instead.
  static const bool ok = is_macos_13_or_newer(MacOSVersion::MACOS_VER_26_4_PLUS);
  return ok;
}

// The host packs dims into std::array<int32_t, 13>; this guards against the
// kernel struct layout drifting from that packing.
static_assert(sizeof(at_gemm::GemmDimsStrided) == 13 * sizeof(int32_t));
static_assert(sizeof(at_gemm::SplitKDims) == 4 * sizeof(int32_t));
static_assert(sizeof(at_gemm::SplitKReduceDims) == 2 * sizeof(int32_t));
static_assert(sizeof(at_gemm::ConvDims) == 3 * sizeof(int32_t));

void mps_gemm(
    const Tensor& A,
    const Tensor& B,
    const Tensor& out,
    const std::optional<Tensor>& self,
    const Scalar& alpha,
    const Scalar& beta,
    at_gemm::GemmEpilogue epi,
    bool force_precise_fp32) {
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

  auto stream = getCurrentMPSStream();

  // GEMV fast path: 2-D rank-1 (M==1 xor N==1), real or integer. Bandwidth-bound
  // kernels that avoid the ~97%-masked GEMM tile. Output must be unit-stride along
  // its length (the kernel writes y[idx]); else fall through to the tiled path.
  if (!batched && ((M == 1) != (N == 1)) && !c10::isComplexType(dt)) {
    const bool m_is_one = (M == 1);
    const int64_t outlen = m_is_one ? N : M;
    const bool out_unit = m_is_one ? (target.stride(c) == 1) : (target.stride(r) == 1);
    if (outlen >= 16 && out_unit) {
      const auto& mat = m_is_one ? rb : ra; // M==1: matrix is B; N==1: matrix is A
      // gemv_t when the output runs along the matrix's columns:
      //   M==1 & !trans_b  or  N==1 & trans_a  -> gemv_t ; otherwise gemv_nt.
      const bool use_t = m_is_one ? !mat.trans : mat.trans;
      const int64_t vec_xs = m_is_one ? A.stride(c) : B.stride(r);
      const Tensor& vmat = mat.view;
      const Tensor& vvec = m_is_one ? A : B;
      int32_t out_stride = 0;
      if (epi == at_gemm::GemmEpilogue::AlphaBeta) {
        out_stride = static_cast<int32_t>(m_is_one ? self_e.stride(c) : self_e.stride(r));
      }
      // GemvDims: {n, K, ld, xs, self_r, self_c}. gemv_t indexes self at (0,n) so
      // the addend step goes in self_c; gemv_nt indexes (row,0) so it goes in self_r.
      const std::array<int32_t, 6> gdims = {
          static_cast<int32_t>(outlen),
          static_cast<int32_t>(K),
          static_cast<int32_t>(mat.ld),
          static_cast<int32_t>(vec_xs),
          use_t ? 0 : out_stride,
          use_t ? out_stride : 0};
      // VEC/NWARPS heuristic (clamped to the matrix's row-stride|offset alignment).
      const int64_t mat_align = mat.ld | vmat.storage_offset();
      const GemvCfg cfg = use_t
          ? pick_gemv_t(dt, outlen, K, mat_align, gemm_use_mpp())
          : pick_gemv_nt(dt, K, mat_align, gemm_use_mpp());
      const std::string fname = use_t ? gemv_t_name(dt_str, cfg.nw, cfg.vec, epi)
                                      : gemv_nt_name(dt_str, cfg.nw, cfg.vec, epi);
      auto pso = lib.getPipelineStateForFunc(fname);
      const NSUInteger tg = static_cast<NSUInteger>(cfg.nw * 32);
      const int64_t ng = use_t ? ((outlen + 32 * cfg.vec - 1) / (32 * cfg.vec))
                               : ((outlen + cfg.nw - 1) / cfg.nw);
      auto run = [&](auto ab_arr) {
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          @autoreleasepool {
            getMPSProfiler().beginProfileKernel(pso, "gemm_gemv", {vmat, vvec});
            auto enc = stream->commandEncoder();
            [enc setComputePipelineState:pso];
            mtl_setArgs(enc, vmat, vvec, target, gdims, self_e, ab_arr);
            [enc dispatchThreadgroups:MTLSizeMake(ng, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            getMPSProfiler().endProfileKernel(pso);
          }
        });
      };
      if (dt == kLong) {
        run(std::array<int64_t, 2>{alpha.toLong(), beta.toLong()});
      } else if (c10::isIntegralType(dt, /*includeBool=*/false)) {
        run(std::array<int32_t, 2>{
            static_cast<int32_t>(alpha.toLong()), static_cast<int32_t>(beta.toLong())});
      } else {
        run(std::array<float, 2>{
            static_cast<float>(alpha.toDouble()), static_cast<float>(beta.toDouble())});
      }
      if (!target.is_same(out)) {
        out.copy_(target);
      }
      return;
    }
  }

  // Integer GEMM: the matrix/tensor units are float-only, so route to the
  // register-tiled int_gemm. Reuses GemmDimsStrided + apply_epilogue; alpha/beta
  // are bound in the accumulate type (opmath_t<DT>: int for <=32-bit, long else).
  if (c10::isIntegralType(dt, /*includeBool=*/false)) {
    const IntTile t = pick_int_tile(M, N, K, dt);
    const std::string fname = int_name(dt_str, t, ra.trans, rb.trans, epi, batched);
    auto pso = lib.getPipelineStateForFunc(fname);
    const int64_t tiles_m = (M + t.BM - 1) / t.BM;
    const int64_t tiles_n = (N + t.BN - 1) / t.BN;
    const NSUInteger tg = static_cast<NSUInteger>(t.TX * t.TY);
    auto run = [&](auto alpha_beta_int) {
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          getMPSProfiler().beginProfileKernel(pso, "gemm_int", {ra.view, rb.view});
          auto enc = stream->commandEncoder();
          [enc setComputePipelineState:pso];
          mtl_setArgs(enc, ra.view, rb.view, target, dims, self_e, alpha_beta_int);
          [enc dispatchThreadgroups:MTLSizeMake(tiles_n, tiles_m, batch)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
          getMPSProfiler().endProfileKernel(pso);
        }
      });
    };
    if (dt == kLong) {
      run(std::array<int64_t, 2>{alpha.toLong(), beta.toLong()});
    } else {
      run(std::array<int32_t, 2>{
          static_cast<int32_t>(alpha.toLong()), static_cast<int32_t>(beta.toLong())});
    }
    if (!target.is_same(out)) {
      out.copy_(target);
    }
    return;
  }

  const std::array<float, 2> alpha_beta = {
      static_cast<float>(alpha.toDouble()), static_cast<float>(beta.toDouble())};

  // matmul2d routing: the mpp kernel handles every resolvable layout - packed,
  // transposed (column-major) and arbitrary leading dim all ride its strided
  // tensor view + matmul2d transpose flags. The only requirement is a row-major
  // (unit-inner-stride) output, which `target` guarantees. Everything else (matmul2d
  // unavailable, or a tiny shape below the tile floor) falls to the portable simd
  // kernel. fp32 defaults to TF32-relaxed (the chosen MPS speed default); set
  // PYTORCH_MPS_PREFER_FP32_PRECISE for exact fp32, except that precise fp32 with a
  // transposed operand has no mpp instantiation and routes to the fp32-exact
  // simd kernel. bf16/fp16 accumulate in fp32, so they are inherently "relaxed".
  const bool use_mpp = gemm_use_mpp();
  static const bool precise_fp32 = c10::utils::has_env("PYTORCH_MPS_PREFER_FP32_PRECISE");
  const bool want_relaxed = (dt != kFloat) || (!precise_fp32 && !force_precise_fp32);
  const bool transposed = ra.trans || rb.trans;
  // Precise fp32 + transposed is the one tensor-unit combo we do not instantiate.
  const bool m5t_has_variant = want_relaxed || !transposed;
  const bool use_m5t =
      use_mpp && m5t_has_variant && M >= 2 && N >= 32 && K >= 64;
  const bool relaxed = want_relaxed;

  if (use_m5t) {
    const bool packed_untrans = !transposed && ra.ld == K && rb.ld == N;
    const GemmPlan plan = choose_gemm_plan(
        dt, dt_str, relaxed, batched, packed_untrans, ra.view, rb.view, M, N, K, batch);
    const bool epi_none = (epi == at_gemm::GemmEpilogue::None);
    const bool out_packed = (ldc == N) && (!batched || target.stride(0) == M * N);
    // split-K / conv only run for a bare (no-epilogue) packed-output mm; addmm or
    // a cached sk/conv plan we cannot use here fall back to the m5t tile.
    if (epi_none && out_packed && plan.backend == GemmPlan::kSplitK) {
      launch_splitk(dt_str, ra.view, rb.view, target, M, N, K,
                    plan.tile.BM, plan.tile.BN, plan.tile.NSG, plan.G);
    } else if (epi_none && out_packed && plan.backend == GemmPlan::kConv) {
      launch_conv(dt_str, ra.view, rb.view, target, M, N, K, plan.BMW, plan.BNO, plan.convNSG);
    } else {
      const TensorTile t = (plan.backend == GemmPlan::kM5T)
          ? plan.tile
          : pick_m5_tensor_tile(M, N, K, dt);
      const std::string fname = m5t_name(dt_str, t, ra.trans, rb.trans, relaxed, epi, batched);
      auto pso = lib.getPipelineStateForFunc(fname);
      const int64_t tiles_m = (M + t.BM - 1) / t.BM;
      const int64_t tiles_n = (N + t.BN - 1) / t.BN;
      const NSUInteger tg = static_cast<NSUInteger>(t.NSG * 32);
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          getMPSProfiler().beginProfileKernel(pso, "mpp_gemm", {ra.view, rb.view});
          auto enc = stream->commandEncoder();
          [enc setComputePipelineState:pso];
          mtl_setArgs(enc, ra.view, rb.view, target, dims, self_e, alpha_beta);
          [enc dispatchThreadgroups:MTLSizeMake(tiles_n, tiles_m, batch)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
          getMPSProfiler().endProfileKernel(pso);
        }
      });
    }
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

void mps_gemm_complex(
    const Tensor& A,
    const Tensor& B,
    const Tensor& out,
    const std::optional<Tensor>& self,
    const Scalar& alpha,
    const Scalar& beta,
    at_gemm::GemmEpilogue epi) {
  if (out.numel() == 0) {
    return;
  }
  const auto cdt = out.scalar_type();
  const auto rdt = (cdt == kComplexFloat) ? kFloat : kHalf;

  // Promote a real operand (torch allows complex @ real), resolve lazy conj/neg
  // views (the kernels read raw storage), and pack for the split kernel.
  auto prep = [&](const Tensor& t) {
    Tensor x = (t.scalar_type() != cdt) ? t.to(cdt) : t;
    return x.resolve_conj().resolve_neg().contiguous();
  };
  Tensor a = prep(A);
  Tensor b = prep(B);

  // Rank-1 (M==1 / N==1): native interleaved-complex GEMV reads the matrix once,
  // vs the four-real-GEMM decomposition below. Both operands are now contiguous.
  if (a.dim() == 2 && b.dim() == 2 &&
      launch_cgemv_complex(a, b, out, self, alpha, beta, epi)) {
    return;
  }

  const auto ropt = out.options().dtype(rdt);
  Tensor ar = at::empty(a.sizes(), ropt);
  Tensor ai = at::empty(a.sizes(), ropt);
  Tensor br = at::empty(b.sizes(), ropt);
  Tensor bi = at::empty(b.sizes(), ropt);
  launch_complex_split(a, ar, ai);
  launch_complex_split(b, br, bi);

  // Four real products on the real/imag planes. complex64 forces full fp32 so the
  // result keeps the precision of the old MPSGraph complex path.
  const bool precise = (cdt == kComplexFloat);
  Tensor P = at::empty(out.sizes(), ropt);
  Tensor Q = at::empty(out.sizes(), ropt);
  Tensor S = at::empty(out.sizes(), ropt);
  Tensor T = at::empty(out.sizes(), ropt);
  mps_gemm(ar, br, P, std::nullopt, 1, 0, at_gemm::GemmEpilogue::None, precise);
  mps_gemm(ai, bi, Q, std::nullopt, 1, 0, at_gemm::GemmEpilogue::None, precise);
  mps_gemm(ar, bi, S, std::nullopt, 1, 0, at_gemm::GemmEpilogue::None, precise);
  mps_gemm(ai, br, T, std::nullopt, 1, 0, at_gemm::GemmEpilogue::None, precise);

  // Recombine into a fresh contiguous complex tensor (the combine kernel writes a
  // flat buffer; a temp also avoids aliasing with `self` during the epilogue).
  Tensor combined = at::empty(out.sizes(), out.options());
  launch_complex_combine(P, Q, S, T, combined);

  if (epi == at_gemm::GemmEpilogue::AlphaBeta) {
    combined.mul_(alpha);
    const auto bv = beta.toComplexDouble();
    if (bv.real() != 0.0 || bv.imag() != 0.0) {
      TORCH_INTERNAL_ASSERT(self.has_value());
      combined.add_(self->expand_as(out), beta);
    }
  }
  out.copy_(combined);
}

} // namespace at::native::mps
