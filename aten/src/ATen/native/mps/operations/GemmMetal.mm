//  Hand-written Metal GEMM dispatcher for the MPS backend. Ports the metalBLAS
//  dispatch brain (resolve_inputs, tile pickers, tensor-unit gate, autotuner) to
//  host C++, replacing the MPSGraph matmul path for float/half/bfloat.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/operations/GemmMetal.h>

#include <ATen/Context.h>
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

// metalBLAS _resolve_inputs, per matrix (rdim/cdim = the logical row/col dims).
// Row-major keeps the view (ld = row stride); column-major is flagged transposed;
// anything else is made contiguous.
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

std::string simd_name(const std::string& dt,
                      SimdTile t,
                      bool trans_a,
                      bool trans_b,
                      at_gemm::GemmEpilogue epi,
                      bool batched) {
  return fmt::format("gemm_simd_{}_{}_{}_{}_{}_{}_ta{}_tb{}_{}_{}",
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

// (BM, BN, NSG) for the mpp (matmul2d) kernel.
struct TensorTile {
  int BM, BN, NSG;
};

TensorTile pick_mpp_tile(int64_t M, int64_t N, int64_t K, c10::ScalarType dt) {
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

std::string mpp_name(const std::string& dt,
                     TensorTile t,
                     bool trans_a,
                     bool trans_b,
                     bool relaxed,
                     at_gemm::GemmEpilogue epi,
                     bool batched) {
  return fmt::format("gemm_mpp_{}_{}_{}_{}_ta{}_tb{}_{}_{}_{}",
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
  return fmt::format("gemv_t_{}_{}_{}_{}", dt, nw, vec, epi == at_gemm::GemmEpilogue::AlphaBeta ? "ab" : "none");
}

std::string gemv_nt_name(const std::string& dt, int nw, int vec, at_gemm::GemmEpilogue epi) {
  return fmt::format("gemv_nt_{}_{}_{}_{}", dt, nw, vec, epi == at_gemm::GemmEpilogue::AlphaBeta ? "ab" : "none");
}

// c2 = float2 (complex64) / half2 (complex32).
std::string cgemv_t_name(const std::string& c2, int nw, at_gemm::GemmEpilogue epi) {
  return fmt::format("cgemv_t_{}_{}_{}", c2, nw, epi == at_gemm::GemmEpilogue::AlphaBeta ? "ab" : "none");
}

std::string cgemv_nt_name(const std::string& c2, int nw, at_gemm::GemmEpilogue epi) {
  return fmt::format("cgemv_nt_{}_{}_{}", c2, nw, epi == at_gemm::GemmEpilogue::AlphaBeta ? "ab" : "none");
}

// (NWARPS, VEC) for a GEMV launch (metalBLAS _gemv_pick / _gemv_nt_pick). VEC scales
// with the output width and is clamped to the matrix row-stride|offset alignment.
// Every returned (nw, vec) has a matching instantiation in Gemm.metal (MB_GEMV_*).
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

GemvCfg pick_gemv_t(c10::ScalarType dt, int64_t outlen, int64_t K, int64_t align, bool tensor_unit) {
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

// gemv_bt launch params. mrows is the padded row capacity {2,4,8,16}; trans_b
// marks a column-major B. Keep in sync with the instantiations in Gemm.metal.
struct GemvBtSpec {
  int mrows, vec, nwarps, ncols;
  bool trans_b;
  bool valid;
};

// Round M up to the instantiated row capacity.
int gemv_bt_cap(int64_t M) {
  if (M <= 2) {
    return 2;
  }
  if (M <= 4) {
    return 4;
  }
  if (M <= 8) {
    return 8;
  }
  return 16;
}

// NWARPS derived from MROWS*VEC. MUST match gemv_bt_nwarps() in gemm_gemv_bt.h: it
// caps the row-major partials tile (NWARPS*MROWS*32*VEC floats) in threadgroup mem.
int gemv_bt_nwarps(int mrows, int vec) {
  return (mrows * vec <= 24) ? 8 : 4;
}

// Heuristic gemv_bt config (metalBLAS _gemv_bt_specs). `align` is the OR of the
// strides VEC loads must divide; invalid when VEC clamps below the column min of 2.
GemvBtSpec pick_gemv_bt(int64_t M, int64_t N, int64_t K, bool trans_b, int64_t align) {
  GemvBtSpec s{};
  s.valid = false;
  const int cap = gemv_bt_cap(M);
  if (trans_b) {
    int vec = clamp_vec((K >= 2048) ? 8 : (K >= 512 ? 4 : 2), align);
    if (vec < 2) {
      return s; // the column-major path vectorizes both X and B over K
    }
    int ncols = 1;
    if (M >= 6) { // below ~6 rows X is not the bottleneck; cap*ncols<=48 registers
      if (cap * 4 <= 48) {
        ncols = 4;
      } else if (cap * 2 <= 48) {
        ncols = 2;
      }
    }
    s = {cap, vec, gemv_bt_nwarps(cap, vec), ncols, true, true};
    return s;
  }
  int vec = clamp_vec((N >= 4096) ? 4 : (N >= 256 ? 2 : 1), align);
  while (vec > 1 && cap * vec > 32) { // accumulator register cap (matches instantiations)
    vec >>= 1;
  }
  s = {cap, vec, gemv_bt_nwarps(cap, vec), 1, false, true};
  return s;
}

std::string gemv_bt_name(const std::string& dt, const GemvBtSpec& s, at_gemm::GemmEpilogue epi) {
  const char* en = (epi == at_gemm::GemmEpilogue::AlphaBeta) ? "ab" : "none";
  if (s.trans_b) {
    return fmt::format("gemv_bt_t_{}_{}_{}_{}_{}", dt, s.mrows, s.vec, s.ncols, en);
  }
  return fmt::format("gemv_bt_{}_{}_{}_{}", dt, s.mrows, s.vec, en);
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

std::string int_name(const std::string& dt,
                     IntTile t,
                     bool trans_a,
                     bool trans_b,
                     at_gemm::GemmEpilogue epi,
                     bool batched) {
  return fmt::format("gemm_int_{}_{}_{}_{}_{}_{}_ta{}_tb{}_{}_{}",
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
void launch_complex_combine(const Tensor& P, const Tensor& Q, const Tensor& S, const Tensor& T, const Tensor& dst) {
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

// Native interleaved-complex rank-1 GEMV (M==1 / N==1): reads the matrix once as C2
// (vs the four-real-GEMM decomposition). a, b are already contiguous of the out
// dtype. Returns true when handled, false to fall back to the decomposition.
bool launch_cgemv_complex(const Tensor& a,
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
  const std::array<int32_t, 6> gdims = {static_cast<int32_t>(outlen),
                                        static_cast<int32_t>(K),
                                        static_cast<int32_t>(ld),
                                        /*xs=*/1,
                                        m_is_one ? 0 : self_r, // cgemv_nt indexes self at (row, 0)
                                        m_is_one ? self_c : 0}; // cgemv_t indexes self at (0, n)

  const int nw = m_is_one ? 8 : 4;
  const std::string fname = m_is_one ? cgemv_t_name(c2, nw, epi) : cgemv_nt_name(c2, nw, epi);
  auto pso = lib.getPipelineStateForFunc(fname);
  const auto av = alpha.toComplexDouble();
  const auto bv = beta.toComplexDouble();
  const std::array<float, 4> ab = {static_cast<float>(av.real()),
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
      [enc dispatchThreadgroups:MTLSizeMake(ng, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
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
// Ambiguous regimes probe a candidate list once on scratch, cached by (dtype, M, N, K,
// batched); confident regimes skip it. Disable with PYTORCH_MPS_GEMM_AUTOTUNE=0.
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

std::vector<TensorTile> with_primary(TensorTile primary, std::vector<TensorTile> extra) {
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
// instantiated untransposed mpp tile (Gemm.metal MB_MPP_* + MB_MPP_UNTRANS).
std::pair<std::vector<TensorTile>, double> mpp_tensor_tile_candidates(int64_t M,
                                                                      int64_t N,
                                                                      int64_t K,
                                                                      c10::ScalarType dt) {
  TensorTile primary = pick_mpp_tile(M, N, K, dt);
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
    if (N == 64 && M >= 4096 && K >= 2048) {
      return {{TensorTile{32, 64, 2}}, kAutotuneMargin};
    }
    if (N == 128 && M >= 4096 && K >= 2048) {
      const TensorTile thin_n_tile = M >= 8192 ? TensorTile{256, 128, 32} : TensorTile{512, 64, 32};
      return {{thin_n_tile}, kAutotuneMargin};
    }
    if (N == 256 && M >= 4096 && K >= 2048) {
      return {{primary}, kAutotuneMargin};
    }
    std::vector<TensorTile> extra = {
        {128, 32, 2}, {256, 32, 4}, {32, 128, 2}, {32, 256, 4}, {64, 64, 2}, {64, 32, 2}, {32, 64, 2}};
    if (N == 128 && M >= 1024) {
      extra.push_back({512, 64, 32});
      if (M >= 8192) {
        extra.push_back({256, 128, 32});
      }
    }
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
    return pick_mpp_tile(M, N, K, dt);
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
      {64, 64, 2}, {64, 64, 4}, {32, 64, 2}, {32, 128, 4}, {64, 128, 4}, {128, 64, 4}, {128, 32, 4}, {128, 128, 8}};
  std::vector<TensorTile> extra;
  for (auto t : kBmmTiles) {
    if (t.BM <= 2 * M && t.BN <= 2 * N) {
      extra.push_back(t);
    }
  }
  return with_primary(pick_bmm_tile(M, N, K, dt), extra);
}

// A resolved launch plan: an mpp tile, a split-K / 1x1-conv candidate
// (deep-K / thin-N regimes the autotuner verifies against the mpp result),
// or a thin-M gemv_bt spec (the autotuner times it against the mpp tiles).
struct GemmPlan {
  enum Backend { kMPP, kSplitK, kConv, kGemvBt } backend = kMPP;
  TensorTile tile{64, 64, 2}; // mpp tile, or split-K (BM, BN, NSG)
  int G = 0; // split-K chunks
  int BMW = 0, BNO = 0, convNSG = 0; // conv
  GemvBtSpec gbt{}; // valid only when backend == kGemvBt
};

// metalBLAS _is_splitk_regime / _is_conv_regime + spec families (low precision).
struct SplitKSpec {
  int BM, BN, NSG, G;
};
struct ConvSpec {
  int BMW, BNO, NSG;
};

bool is_splitk_regime(int64_t M, int64_t N, int64_t K, c10::ScalarType dt) {
  return dt != kFloat && K >= 2048 && std::min(M, N) > 128 && M * N <= 1500000 &&
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
// precompiled K set (the AOT analog of metalBLAS's JIT-per-K); other K -> mpp.
bool conv_k_supported(int64_t K) {
  return K == 512 || K == 1024 || K == 2048 || K == 4096;
}
bool is_conv_regime(int64_t M, int64_t N, int64_t K, c10::ScalarType dt) {
  const bool use_n64_conv = N < 64 || M > 4096;
  return dt != kFloat && N <= 64 && use_n64_conv && N % 32 == 0 && M >= 512 && conv_k_supported(K);
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
void launch_splitk(const std::string& dt_str,
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
  auto pso_g = lib.getPipelineStateForFunc(fmt::format("splitk_gemm_{}_{}_{}_{}", dt_str, BM, BN, NSG));
  auto pso_r = lib.getPipelineStateForFunc("splitk_reduce_" + dt_str);
  const int64_t tiles_m = (M + BM - 1) / BM;
  const int64_t tiles_n = (N + BN - 1) / BN;
  const std::array<int32_t, 4> sk = {static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K), kchunk};
  const std::array<int32_t, 2> rd = {static_cast<int32_t>(M * N), planes};
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso_g, "splitk_gemm", {A, B});
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso_g];
      mtl_setArgs(enc, A, B, Cp, sk);
      [enc dispatchThreadgroups:MTLSizeMake(tiles_n, tiles_m, G) threadsPerThreadgroup:MTLSizeMake(NSG * 32, 1, 1)];
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
void launch_conv(const std::string& dt_str,
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
  auto pso = lib.getPipelineStateForFunc(fmt::format("conv1x1_gemm_{}_{}_{}_{}_{}", dt_str, BMW, BNO, NSG, K));
  const int64_t tiles_o = (N + BNO - 1) / BNO;
  const int64_t tiles_w = (M + BMW - 1) / BMW;
  const std::array<int32_t, 3> cd = {static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K)};
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "conv1x1_gemm", {A, B});
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, A, B, out, cd);
      [enc dispatchThreadgroups:MTLSizeMake(tiles_o, tiles_w, 1) threadsPerThreadgroup:MTLSizeMake(NSG * 32, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

// Thin-M gemv_bt launch (row- or column-major B per spec.trans_b); encode-only.
// Shared by the autotuner probe and the final dispatch, so the timed kernel is
// exactly the one that runs. Caller supplies the GemvBtDims fields.
void launch_gemv_bt(const std::string& dt_str,
                    const GemvBtSpec& spec,
                    at_gemm::GemmEpilogue epi,
                    const Tensor& B,
                    const Tensor& X,
                    const Tensor& out,
                    const Tensor& self,
                    int64_t M,
                    int64_t N,
                    int64_t K,
                    int32_t ldb,
                    int32_t ldx,
                    int32_t ldy,
                    int32_t self_r,
                    int32_t self_c,
                    int32_t batch_b,
                    int32_t batch_x,
                    int32_t batch_y,
                    int32_t batch_self,
                    const std::array<float, 2>& alpha_beta,
                    int64_t batch) {
  // Field order must match at_gemm::GemvBtDims.
  const std::array<int32_t, 12> bd = {static_cast<int32_t>(M),
                                      static_cast<int32_t>(N),
                                      static_cast<int32_t>(K),
                                      ldb,
                                      ldx,
                                      ldy,
                                      self_r,
                                      self_c,
                                      batch_b,
                                      batch_x,
                                      batch_y,
                                      batch_self};
  auto pso = lib.getPipelineStateForFunc(gemv_bt_name(dt_str, spec, epi));
  const int per = spec.trans_b ? (spec.nwarps * spec.ncols) : (32 * spec.vec);
  const int64_t ng = (N + per - 1) / per;
  const NSUInteger tg = static_cast<NSUInteger>(spec.nwarps * 32);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "gemm_gemv_bt", {B, X});
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, B, X, out, bd, self, alpha_beta);
      [enc dispatchThreadgroups:MTLSizeMake(ng, 1, batch) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
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

// Time each candidate (best-of-reps min) on scratch; candidate 0 (heuristic mpp) is
// kept unless beaten by > margin. split-K / conv candidates are verified against the
// mpp reference before joining the probe (metalBLAS _autotune_mppt).
GemmPlan autotune_plan(const std::string& dt_str,
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
                       const std::vector<ConvSpec>& conv_specs,
                       const GemvBtSpec& gbt) {
  auto stream = getCurrentMPSStream();
  Tensor scratch = batched ? at::empty({batch, M, N}, Av.options()) : at::empty({M, N}, Av.options());
  // Packed (ta0 tb0) dims for the scratch (lda=K, ldb=N, ldc=N).
  const std::array<int32_t, 13> dims = {static_cast<int32_t>(M),
                                        static_cast<int32_t>(N),
                                        static_cast<int32_t>(K),
                                        static_cast<int32_t>(K),
                                        static_cast<int32_t>(N),
                                        static_cast<int32_t>(N),
                                        0,
                                        0,
                                        0,
                                        batched ? static_cast<int32_t>(M * K) : 0,
                                        batched ? static_cast<int32_t>(K * N) : 0,
                                        batched ? static_cast<int32_t>(M * N) : 0,
                                        0};
  const std::array<float, 2> ab = {1.0f, 0.0f};

  auto enqueue_mpp = [&](TensorTile t, const Tensor& o) {
    const std::string fname = mpp_name(dt_str, t, false, false, relaxed, at_gemm::GemmEpilogue::None, batched);
    auto pso = lib.getPipelineStateForFunc(fname);
    const int64_t tiles_m = (M + t.BM - 1) / t.BM;
    const int64_t tiles_n = (N + t.BN - 1) / t.BN;
    const NSUInteger tg = static_cast<NSUInteger>(t.NSG * 32);
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto enc = stream->commandEncoder();
        [enc setComputePipelineState:pso];
        mtl_setArgs(enc, Av, Bv, o, dims, Av, ab);
        [enc dispatchThreadgroups:MTLSizeMake(tiles_n, tiles_m, batch) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      }
    });
  };

  struct Cand {
    GemmPlan plan;
    std::function<void(const Tensor&)> run;
  };
  std::vector<Cand> cand;
  for (auto t : cands) {
    cand.push_back({GemmPlan{GemmPlan::kMPP, t}, [&, t](const Tensor& o) { enqueue_mpp(t, o); }});
  }
  // Thin-M gemv_bt joins the probe as a peer of the mpp tiles: correctness is
  // already proven against mpp, so unlike split-K/conv it needs no match gate.
  if (gbt.valid) {
    GemmPlan p;
    p.backend = GemmPlan::kGemvBt;
    p.gbt = gbt;
    cand.push_back({p, [&, gbt](const Tensor& o) {
                      launch_gemv_bt(dt_str,
                                     gbt,
                                     at_gemm::GemmEpilogue::None,
                                     Bv,
                                     Av,
                                     o,
                                     Av,
                                     M,
                                     N,
                                     K,
                                     static_cast<int32_t>(N),
                                     static_cast<int32_t>(K),
                                     static_cast<int32_t>(N),
                                     0,
                                     0,
                                     batched ? static_cast<int32_t>(K * N) : 0,
                                     batched ? static_cast<int32_t>(M * K) : 0,
                                     batched ? static_cast<int32_t>(M * N) : 0,
                                     0,
                                     ab,
                                     batch);
                    }});
  }

  // split-K / conv candidates: included only when their result matches the mpp
  // reference everywhere, via an allclose-style per-element gate (not a global max-
  // abs, which a kernel wrong only at a few small outputs would slip past).
  if (!sk_specs.empty() || !conv_specs.empty()) {
    Tensor ref = at::empty({M, N}, Av.options());
    cand[0].run(ref);
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
    auto matches = [&](const Tensor& o) {
      // both kernels accumulate in fp32, so a correct candidate differs from the
      // mpp reference only by bf16/fp16 rounding (~1 ULP); 5% rtol + small atol.
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
        cand.push_back(
            {p, [&, sp](const Tensor& o) { launch_splitk(dt_str, Av, Bv, o, M, N, K, sp.BM, sp.BN, sp.NSG, sp.G); }});
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
        cand.push_back(
            {p, [&, sp](const Tensor& o) { launch_conv(dt_str, Av, Bv, o, M, N, K, sp.BMW, sp.BNO, sp.NSG); }});
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

  // Bound the probe by wall-time, not FLOPs: thin shapes have tiny FLOPs but slow
  // kernels (a 16x4096x4096 tile ~110us, a high-M gemv_bt ~850us), so the FLOP-scaled
  // iters above would probe them for seconds. Warm once so every PSO compiles, then
  // pilot the compiled kernel to scale warmup/iters per candidate to a few ms each
  // (still capped by the FLOP budget, so genuinely tiny shapes keep full resolution).
  // The warm pass matters: timing the cold first run would fold in the one-time
  // compile, collapse itj to 1, and make the small-shape picks noisy.
  for (auto& c : cand) {
    c.run(scratch);
  }
  stream->synchronize(SyncType::COMMIT_AND_WAIT);

  std::vector<double> pilot(cand.size(), 0.0);
  for (size_t j = 0; j < cand.size(); ++j) {
    auto t0 = std::chrono::steady_clock::now();
    cand[j].run(scratch);
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
    pilot[j] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  auto budget = [](double per_call, double target_s, int cap) {
    return std::max(1, std::min(cap, static_cast<int>(target_s / std::max(per_call, 1e-6))));
  };

  for (size_t j = 0; j < cand.size(); ++j) {
    const int wj = budget(pilot[j], 3e-3, warmup);
    for (int w = 0; w < wj; ++w) {
      cand[j].run(scratch);
    }
  }
  stream->synchronize(SyncType::COMMIT_AND_WAIT);

  std::vector<double> best(cand.size(), 1e30);
  for (int r = 0; r < reps; ++r) {
    for (size_t j = 0; j < cand.size(); ++j) {
      const int itj = budget(pilot[j], 2e-3, iters);
      auto t0 = std::chrono::steady_clock::now();
      for (int it = 0; it < itj; ++it) {
        cand[j].run(scratch);
      }
      stream->synchronize(SyncType::COMMIT_AND_WAIT);
      const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / itj;
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

// Resolve the launch plan: heuristic mpp for transposed / non-packed / autotune-off,
// else the cached autotuned winner (probing once on first sight of the shape).
GemmPlan choose_gemm_plan(c10::ScalarType dt,
                          const std::string& dt_str,
                          bool relaxed,
                          bool batched,
                          bool packed_untrans,
                          const Tensor& Av,
                          const Tensor& Bv,
                          int64_t M,
                          int64_t N,
                          int64_t K,
                          int64_t batch,
                          const GemvBtSpec& gbt) {
  if (!autotune_enabled() || !packed_untrans) {
    GemmPlan p;
    p.tile = batched ? pick_bmm_tile(M, N, K, dt) : pick_mpp_tile(M, N, K, dt);
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
  if (cands.size() > 1 || !sk.empty() || !cv.empty() || gbt.valid) {
    plan = autotune_plan(dt_str, relaxed, batched, Av, Bv, M, N, K, batch, cands, margin, sk, cv, gbt);
  } else {
    plan.backend = GemmPlan::kMPP;
    plan.tile = cands[0];
  }
  {
    std::lock_guard<std::mutex> g(g_tile_mutex);
    g_tile_cache.emplace(key, plan); // first writer wins
  }
  return plan;
}

} // namespace

bool gemm_supported_dtype(c10::ScalarType dt) {
  return dt == kFloat || dt == kHalf || dt == kBFloat16 || c10::isIntegralType(dt, /*includeBool=*/false);
}

bool gemm_use_mpp() {
  // matmul2d (MetalPerformancePrimitives) is gated only on the OS: macOS 26.2+ with
  // no GPU-family requirement (it lowers to the NAX matrix unit where present and to
  // simdgroup execution otherwise).
  static const bool ok = is_macos_13_or_newer(MacOSVersion::MACOS_VER_26_2_PLUS);
  return ok;
}

// The host packs dims into std::array<int32_t, 13>; this guards against the
// kernel struct layout drifting from that packing.
static_assert(sizeof(at_gemm::GemmDimsStrided) == 13 * sizeof(int32_t));
static_assert(sizeof(at_gemm::SplitKDims) == 4 * sizeof(int32_t));
static_assert(sizeof(at_gemm::SplitKReduceDims) == 2 * sizeof(int32_t));
static_assert(sizeof(at_gemm::ConvDims) == 3 * sizeof(int32_t));
static_assert(sizeof(at_gemm::GemvBtDims) == 12 * sizeof(int32_t));

void mps_gemm(const Tensor& A,
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
  const std::array<int32_t, 13> dims = {static_cast<int32_t>(M),
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
      const std::array<int32_t, 6> gdims = {static_cast<int32_t>(outlen),
                                            static_cast<int32_t>(K),
                                            static_cast<int32_t>(mat.ld),
                                            static_cast<int32_t>(vec_xs),
                                            use_t ? 0 : out_stride,
                                            use_t ? out_stride : 0};
      // VEC/NWARPS heuristic (clamped to the matrix's row-stride|offset alignment).
      const int64_t mat_align = mat.ld | vmat.storage_offset();
      const GemvCfg cfg = use_t ? pick_gemv_t(dt, outlen, K, mat_align, gemm_use_mpp())
                                : pick_gemv_nt(dt, K, mat_align, gemm_use_mpp());
      const std::string fname =
          use_t ? gemv_t_name(dt_str, cfg.nw, cfg.vec, epi) : gemv_nt_name(dt_str, cfg.nw, cfg.vec, epi);
      auto pso = lib.getPipelineStateForFunc(fname);
      const NSUInteger tg = static_cast<NSUInteger>(cfg.nw * 32);
      const int64_t ng = use_t ? ((outlen + 32 * cfg.vec - 1) / (32 * cfg.vec)) : ((outlen + cfg.nw - 1) / cfg.nw);
      auto run = [&](auto ab_arr) {
        dispatch_sync_with_rethrow(stream->queue(), ^() {
          @autoreleasepool {
            getMPSProfiler().beginProfileKernel(pso, "gemm_gemv", {vmat, vvec});
            auto enc = stream->commandEncoder();
            [enc setComputePipelineState:pso];
            mtl_setArgs(enc, vmat, vvec, target, gdims, self_e, ab_arr);
            [enc dispatchThreadgroups:MTLSizeMake(ng, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            getMPSProfiler().endProfileKernel(pso);
          }
        });
      };
      if (dt == kLong) {
        run(std::array<int64_t, 2>{alpha.toLong(), beta.toLong()});
      } else if (c10::isIntegralType(dt, /*includeBool=*/false)) {
        run(std::array<int32_t, 2>{static_cast<int32_t>(alpha.toLong()), static_cast<int32_t>(beta.toLong())});
      } else {
        run(std::array<float, 2>{static_cast<float>(alpha.toDouble()), static_cast<float>(beta.toDouble())});
      }
      if (!target.is_same(out)) {
        out.copy_(target);
      }
      return;
    }
  }

  // Transposed-B thin-M GEMV (x @ W.t(), M in 2..16, half/bf16): the lm-head /
  // vocab path the autotuner can't reach (it only probes untransposed tiles). The
  // column-major gemv_bt_t kernel reduces over K with simd_sum and has no high-M
  // cliff, so route it directly. Row-major B instead flows to the autotuner below,
  // which times gemv_bt against the matmul tiles per shape (see choose_gemm_plan).
  if (rb.trans && (dt == kHalf || dt == kBFloat16) && gemm_use_mpp() && !ra.trans && M >= 2 && M <= 16 &&
      K >= 64 && N >= 16 && N <= 262144) {
    const int64_t align = rb.ld | ra.ld | ra.view.storage_offset() | rb.view.storage_offset() |
        (batched ? (ra.view.stride(0) | rb.view.stride(0)) : 0);
    const GemvBtSpec spec = pick_gemv_bt(M, N, K, /*trans_b=*/true, align);
    if (spec.valid) {
      const std::array<float, 2> ab = {static_cast<float>(alpha.toDouble()), static_cast<float>(beta.toDouble())};
      launch_gemv_bt(dt_str,
                     spec,
                     epi,
                     rb.view,
                     ra.view,
                     target,
                     self_e,
                     M,
                     N,
                     K,
                     static_cast<int32_t>(rb.ld),
                     static_cast<int32_t>(ra.ld),
                     static_cast<int32_t>(ldc),
                     self_r,
                     self_c,
                     batched ? static_cast<int32_t>(rb.view.stride(0)) : 0,
                     batched ? static_cast<int32_t>(ra.view.stride(0)) : 0,
                     batched ? static_cast<int32_t>(target.stride(0)) : 0,
                     batch_self,
                     ab,
                     batch);
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
          [enc dispatchThreadgroups:MTLSizeMake(tiles_n, tiles_m, batch) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
          getMPSProfiler().endProfileKernel(pso);
        }
      });
    };
    if (dt == kLong) {
      run(std::array<int64_t, 2>{alpha.toLong(), beta.toLong()});
    } else {
      run(std::array<int32_t, 2>{static_cast<int32_t>(alpha.toLong()), static_cast<int32_t>(beta.toLong())});
    }
    if (!target.is_same(out)) {
      out.copy_(target);
    }
    return;
  }

  const std::array<float, 2> alpha_beta = {static_cast<float>(alpha.toDouble()), static_cast<float>(beta.toDouble())};

  // matmul2d routing: mpp handles every resolvable layout; tiny shapes or no matmul2d
  // fall to simd. fp32 precision follows set_float32_matmul_precision (HIGHEST -> full,
  // HIGH/MEDIUM -> relaxed TF32); force_precise_fp32 (complex sub-GEMMs) forces full.
  const bool use_mpp = gemm_use_mpp();
  const bool relaxed_fp32 =
      !force_precise_fp32 && at::globalContext().float32MatmulPrecision() != at::Float32MatmulPrecision::HIGHEST;
  const bool want_relaxed = (dt != kFloat) || relaxed_fp32;
  const bool transposed = ra.trans || rb.trans;
  // matmul2d takes K as a runtime extent, including K < 64, and outperforms the
  // portable simd fallback for the small-K regimes tested on Apple10+ and earlier.
  const bool use_mpp_gemm = use_mpp && M >= 2 && N >= 32;
  const bool relaxed = want_relaxed;

  if (use_mpp_gemm) {
    const bool packed_untrans = !transposed && ra.ld == K && rb.ld == N;
    // Row-major thin-M (M in 2..16, half/bf16) enters the probe as a candidate
    // against the matmul tiles; the autotuner caches the per-shape winner.
    GemvBtSpec gbt{};
    if (packed_untrans && (dt == kHalf || dt == kBFloat16) && M >= 2 && M <= 16 && K >= 64 && N >= 16) {
      const int64_t align = rb.ld | rb.view.storage_offset() | (batched ? rb.view.stride(0) : 0);
      gbt = pick_gemv_bt(M, N, K, /*trans_b=*/false, align);
    }
    const GemmPlan plan =
        choose_gemm_plan(dt, dt_str, relaxed, batched, packed_untrans, ra.view, rb.view, M, N, K, batch, gbt);
    const bool epi_none = (epi == at_gemm::GemmEpilogue::None);
    const bool out_packed = (ldc == N) && (!batched || target.stride(0) == M * N);
    // split-K / conv only run for a bare (no-epilogue) packed-output mm; addmm or
    // a cached sk/conv plan we cannot use here fall back to the mpp tile.
    if (plan.backend == GemmPlan::kGemvBt) {
      // Launch with this call's spec (alignment matches the actual operands); the
      // cached plan.gbt only records the autotuner's pick on the first shape seen.
      launch_gemv_bt(dt_str,
                     gbt.valid ? gbt : plan.gbt,
                     epi,
                     rb.view,
                     ra.view,
                     target,
                     self_e,
                     M,
                     N,
                     K,
                     static_cast<int32_t>(rb.ld),
                     static_cast<int32_t>(ra.ld),
                     static_cast<int32_t>(ldc),
                     self_r,
                     self_c,
                     batched ? static_cast<int32_t>(rb.view.stride(0)) : 0,
                     batched ? static_cast<int32_t>(ra.view.stride(0)) : 0,
                     batched ? static_cast<int32_t>(target.stride(0)) : 0,
                     batch_self,
                     alpha_beta,
                     batch);
    } else if (epi_none && out_packed && plan.backend == GemmPlan::kSplitK) {
      launch_splitk(dt_str, ra.view, rb.view, target, M, N, K, plan.tile.BM, plan.tile.BN, plan.tile.NSG, plan.G);
    } else if (epi_none && out_packed && plan.backend == GemmPlan::kConv) {
      launch_conv(dt_str, ra.view, rb.view, target, M, N, K, plan.BMW, plan.BNO, plan.convNSG);
    } else {
      const TensorTile t = (plan.backend == GemmPlan::kMPP) ? plan.tile : pick_mpp_tile(M, N, K, dt);
      const std::string fname = mpp_name(dt_str, t, ra.trans, rb.trans, relaxed, epi, batched);
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
          [enc dispatchThreadgroups:MTLSizeMake(tiles_n, tiles_m, batch) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
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
        [enc dispatchThreadgroups:MTLSizeMake(tiles_n, tiles_m, batch) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        getMPSProfiler().endProfileKernel(pso);
      }
    });
  }

  if (!target.is_same(out)) {
    out.copy_(target);
  }
}

void mps_gemm_complex(const Tensor& A,
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
  if (a.dim() == 2 && b.dim() == 2 && launch_cgemv_complex(a, b, out, self, alpha, beta, epi)) {
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
