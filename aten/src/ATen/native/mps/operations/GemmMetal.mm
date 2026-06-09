#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/operations/GemmMetal.h>

#include <ATen/Context.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#include <array>

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

// Kept as a view when one axis is unit-stride and the other's stride is a valid ld
// (>= extent): col unit-stride -> row-major, row unit-stride -> transposed; else copy.
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

// Bounds are checked in-kernel, so an oversized tile is correct (just wasteful);
// the heuristic keeps enough tiles resident to fill the cores.
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

// Packed untransposed path: thin-N large-M (narrow output projections) wants a
// narrow-BN tile; square 64x64 leaves cores idle on width <= 128. M5 sweeps.
TensorTile pick_mpp_tile_packed(int64_t M, int64_t N, int64_t K, c10::ScalarType dt) {
  if (dt != kFloat && M >= 1024) {
    if (N <= 64) {
      return {64, 32, 2};
    }
    if (N <= 128) {
      return {128, 32, 2};
    }
  }
  return pick_mpp_tile(M, N, K, dt);
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

// (NWARPS, VEC) for a GEMV launch. VEC scales with the output width, clamped to the
// row-stride|offset alignment. Every (nw, vec) has an instantiation in Gemm.metal.
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
    // Integer gemv_t: NW=8 always; VEC by element width (char/uchar 8, short 4,
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
    // Integer gemv_nt: char/uchar (VEC8,NW4), short (VEC4,NW4), int (VEC4,NW8),
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

// Heuristic gemv_bt config. `align` is the OR of the strides VEC loads must divide;
// invalid when VEC clamps below the column min of 2.
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

// {BM, BN, NSG, G} for split-K: G threadgroups each accumulate a K/G chunk into an
// fp32 plane, reduced after. Tiny-output deep-K shapes starve matmul2d for tiles.
struct SplitKSpec {
  int BM, BN, NSG, G;
  bool valid;
};

// Tiny square output (both <= 256) + deep K: matmul2d has too few tiles to fill the
// cores, so split-K wins. Packed + low-precision only (planes accumulate in fp32).
bool is_splitk_regime(int64_t M, int64_t N, int64_t K, c10::ScalarType dt) {
  return dt != kFloat && M <= 256 && N <= 256 && std::min(M, N) >= 128 && K >= 2048;
}

// Fixed M5 winner {32,64,2} (1.3-2.3x). G=4 needs K/4 a multiple of 16 (K % 64 == 0),
// else G=2 (K % 32 == 0), else no aligned split -> invalid, caller uses matmul2d.
SplitKSpec pick_splitk(int64_t K) {
  if (K % 64 == 0) {
    return {32, 64, 2, 4, true};
  }
  if (K % 32 == 0) {
    return {32, 64, 2, 2, true};
  }
  return {0, 0, 0, 0, false};
}

// {BMW, BNO, NSG} for conv1x1 (1x1-conv-shaped GEMM): tall M, very narrow N. The kernel
// name bakes in N (=BNO) and K, so both are gated to instantiated values.
struct ConvSpec {
  int BMW, BNO, NSG;
  bool valid;
};

// conv1x1 is instantiated only for these K (K is a template constant in the kernel).
bool conv_k_supported(int64_t K) {
  return K == 512 || K == 1024 || K == 2048 || K == 4096;
}

// Tall (M >= 512), very narrow N (32 or 64). N=64 needs M > 4096 (else the matmul tile
// already fills). Packed + low precision; K must be an instantiated value.
bool is_conv_regime(int64_t M, int64_t N, int64_t K, c10::ScalarType dt) {
  return dt != kFloat && N <= 64 && (N < 64 || M > 4096) && N % 32 == 0 && M >= 512 && conv_k_supported(K);
}

// M5 winner: BMW=64, NSG=4 (BNO = N). Needs M % 64 == 0 for the row tiling, else invalid.
ConvSpec pick_conv(int64_t M, int64_t N) {
  if (M % 64 != 0) {
    return {0, 0, 0, false};
  }
  return {64, static_cast<int>(N), 4, true};
}

// (BM, BN, BK, TX, TY) for the register-tiled int_gemm kernel.
struct IntTile {
  int BM, BN, BK, TX, TY;
};

// M5 Pro sweeps. Every tile must have a matching instantiation in Gemm.metal.
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
// vs the four-real-GEMM decomposition. a, b already contiguous; false -> fall back.
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

// The batch already fills the cores, so bmm prefers a bigger tile (more K reuse)
// + NSG=4 over the 2-D heuristic's tiny tiles.
TensorTile pick_bmm_tile(int64_t M, int64_t N, int64_t K, c10::ScalarType dt) {
  if (M == 1 || N == 1) {
    return pick_mpp_tile(M, N, K, dt);
  }
  const int64_t mx = std::max(M, N), mn = std::min(M, N);
  if (K <= 128 && mx >= 512) {
    return {32, 128, 4};
  }
  // Thin-M wide-N bmm wants a wide-BN tile (the batch fills the cores); a square mn<=32
  // tile starves the long N dim. M5 sweeps.
  if (N >= 512 && M <= 32) {
    return {32, 128, 4};
  }
  if (N >= 512 && M <= 64) {
    return {64, 128, 4};
  }
  if (mn <= 32) {
    return {32, 32, 4};
  }
  if (mx <= 256) {
    return {64, 64, 4};
  }
  return {64, 64, 2};
}

// Thin-M gemv_bt launch (row- or column-major B per spec.trans_b); encode-only.
// Caller supplies the GemvBtDims fields.
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

// Split-K GEMM: G threadgroups each accumulate a K/G chunk into an fp32 plane, then a
// reduction pass sums the planes into out. Packed inputs only, no epilogue (out = A@B),
// so the dispatcher gates it on the None epilogue.
void launch_splitk(const std::string& dt_str,
                   const Tensor& A,
                   const Tensor& B,
                   const Tensor& out,
                   int64_t M,
                   int64_t N,
                   int64_t K,
                   const SplitKSpec& s) {
  const int kchunk = static_cast<int>(K / s.G);
  Tensor Cp = at::empty({s.G, M, N}, out.options().dtype(kFloat));
  auto stream = getCurrentMPSStream();
  auto pso_g = lib.getPipelineStateForFunc(fmt::format("splitk_gemm_{}_{}_{}_{}", dt_str, s.BM, s.BN, s.NSG));
  auto pso_r = lib.getPipelineStateForFunc("splitk_reduce_" + dt_str);
  const int64_t tiles_m = (M + s.BM - 1) / s.BM;
  const int64_t tiles_n = (N + s.BN - 1) / s.BN;
  const std::array<int32_t, 4> sk = {static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K), kchunk};
  const std::array<int32_t, 2> rd = {static_cast<int32_t>(M * N), s.G};
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso_g, "splitk_gemm", {A, B});
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso_g];
      mtl_setArgs(enc, A, B, Cp, sk);
      [enc dispatchThreadgroups:MTLSizeMake(tiles_n, tiles_m, s.G) threadsPerThreadgroup:MTLSizeMake(s.NSG * 32, 1, 1)];
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

// conv1x1 GEMM: 1x1-conv-shaped kernel for tall, narrow-N projections. Packed inputs,
// no epilogue (out = A@B), gated on the None epilogue like split-K.
void launch_conv(const std::string& dt_str,
                 const Tensor& A,
                 const Tensor& B,
                 const Tensor& out,
                 int64_t M,
                 int64_t N,
                 int64_t K,
                 const ConvSpec& s) {
  auto stream = getCurrentMPSStream();
  auto pso = lib.getPipelineStateForFunc(fmt::format("conv1x1_gemm_{}_{}_{}_{}_{}", dt_str, s.BMW, s.BNO, s.NSG, K));
  const int64_t tiles_o = (N + s.BNO - 1) / s.BNO;
  const int64_t tiles_w = (M + s.BMW - 1) / s.BMW;
  const std::array<int32_t, 3> cd = {static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K)};
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "conv1x1_gemm", {A, B});
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, A, B, out, cd);
      [enc dispatchThreadgroups:MTLSizeMake(tiles_o, tiles_w, 1) threadsPerThreadgroup:MTLSizeMake(s.NSG * 32, 1, 1)];
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

} // namespace

bool gemm_supported_dtype(c10::ScalarType dt) {
  return dt == kFloat || dt == kHalf || dt == kBFloat16 || c10::isIntegralType(dt, /*includeBool=*/false);
}

bool gemm_use_mpp() {
  // matmul2d (MetalPerformancePrimitives) is gated only on the OS (macOS 26.2+); it
  // lowers to the NAX matrix unit where present, simdgroup execution otherwise.
  static const bool ok = is_macos_13_or_newer(MacOSVersion::MACOS_VER_26_2_PLUS);
  return ok;
}

// The host packs dims into std::array<int32_t, 13>; this guards against the
// kernel struct layout drifting from that packing.
static_assert(sizeof(at_gemm::GemmDimsStrided) == 13 * sizeof(int32_t));
static_assert(sizeof(at_gemm::GemvBtDims) == 12 * sizeof(int32_t));

namespace {

// Apple silicon generation, for per-gen heuristic tuning. Unknown -> M5 (the default).
enum class GpuGen : uint8_t { M1, M2, M3, M4, M5 };

GpuGen gemm_gpu_gen() {
  static const GpuGen gen = [] {
    auto dev = MPSDevice::getInstance()->device();
    if (dev != nil) {
      const std::string name([[dev name] UTF8String]); // e.g. "Apple M5 Pro"
      if (name.find("M1") != std::string::npos) {
        return GpuGen::M1;
      }
      if (name.find("M2") != std::string::npos) {
        return GpuGen::M2;
      }
      if (name.find("M3") != std::string::npos) {
        return GpuGen::M3;
      }
      if (name.find("M4") != std::string::npos) {
        return GpuGen::M4;
      }
    }
    return GpuGen::M5;
  }();
  return gen;
}

// Per-gen dispatch policy: the contiguify gate plus this gen's tile/config pickers.
// The single place to tune per silicon; add a `case` in gemm_tuning() below.
struct GemmTuning {
  // Pack a column-major operand to row-major when K is thin AND the output is large:
  // the strided read, re-incurred across many output tiles, amortizes a one-time copy.
  int64_t contig_max_k;
  int64_t contig_min_mn;
  // Tile/config pickers (signatures match the corresponding free functions above).
  SimdTile (*pick_simd)(int64_t, int64_t, int64_t);
  TensorTile (*pick_mpp)(int64_t, int64_t, int64_t, c10::ScalarType);
  TensorTile (*pick_mpp_packed)(int64_t, int64_t, int64_t, c10::ScalarType);
  TensorTile (*pick_bmm)(int64_t, int64_t, int64_t, c10::ScalarType);
  IntTile (*pick_int)(int64_t, int64_t, int64_t, c10::ScalarType);
  GemvCfg (*pick_gemv_t)(c10::ScalarType, int64_t, int64_t, int64_t, bool);
  GemvCfg (*pick_gemv_nt)(c10::ScalarType, int64_t, int64_t, bool);
  GemvBtSpec (*pick_gemv_bt)(int64_t, int64_t, int64_t, bool, int64_t);
};

const GemmTuning& gemm_tuning(GpuGen gen) {
  // M5 Pro sweeps (the default for every gen until swept on its own silicon).
  static const GemmTuning kM5{/*contig_max_k=*/256,
                              /*contig_min_mn=*/2048,
                              &pick_simd_tile,
                              &pick_mpp_tile,
                              &pick_mpp_tile_packed,
                              &pick_bmm_tile,
                              &pick_int_tile,
                              &pick_gemv_t,
                              &pick_gemv_nt,
                              &pick_gemv_bt};
  switch (gen) {
    default:
      return kM5; // M1-M4 fall back to M5 until swept on that silicon.
  }
}

// One GEMM's complete dispatch decision. Only the field(s) for `path` are meaningful.
struct GemmPlan {
  enum class Path : uint8_t { Gemv, GemvBt, Int, Mpp, Simd, SplitK, Conv };
  Path path = Path::Mpp;
  bool contiguify_a = false; // pack a column-major A to row-major before launch
  bool contiguify_b = false;
  bool relaxed = false; // fp32: TF32-relaxed matmul2d (vs full precision)
  bool gemv_use_t = false; // Gemv: gemv_t vs gemv_nt
  GemvCfg gemv{}; // Gemv
  GemvBtSpec gemv_bt{}; // GemvBt (row- or column-major B per gemv_bt.trans_b)
  IntTile int_tile{}; // Int
  TensorTile mpp_tile{}; // Mpp
  SimdTile simd_tile{}; // Simd
  SplitKSpec splitk{}; // SplitK
  ConvSpec conv{}; // Conv
};

// THE dispatch heuristic: shapes/strides/dtype/gen -> plan. Routing order is rank-1
// GEMV -> transposed-B thin-M GEMV -> integer -> matmul2d|simd. Reads only metadata.
GemmPlan gemm_plan(GpuGen gen,
                   const Tensor& A,
                   const Tensor& B,
                   const Tensor& target,
                   const Resolved& a,
                   const Resolved& b,
                   int64_t M,
                   int64_t N,
                   int64_t K,
                   int64_t r,
                   int64_t c,
                   bool batched,
                   c10::ScalarType dt,
                   bool force_precise_fp32,
                   at_gemm::GemmEpilogue epi) {
  GemmPlan p;
  const bool relaxed_fp32 =
      !force_precise_fp32 && at::globalContext().float32MatmulPrecision() != at::Float32MatmulPrecision::HIGHEST;
  p.relaxed = (dt != kFloat) || relaxed_fp32;

  const bool use_mpp = gemm_use_mpp();
  const GemmTuning& tune = gemm_tuning(gen);

  // 1. Rank-1 real/integer GEMV (M==1 xor N==1), output unit-stride along its length.
  if (!batched && ((M == 1) != (N == 1)) && !c10::isComplexType(dt)) {
    const bool m_is_one = (M == 1);
    const int64_t outlen = m_is_one ? N : M;
    const bool out_unit = m_is_one ? (target.stride(c) == 1) : (target.stride(r) == 1);
    if (outlen >= 16 && out_unit) {
      const Resolved& mat = m_is_one ? b : a; // M==1: matrix is B; N==1: matrix is A
      // gemv_t when the output runs along the matrix's columns; else gemv_nt.
      p.gemv_use_t = m_is_one ? !mat.trans : mat.trans;
      const int64_t align = mat.ld | mat.view.storage_offset();
      p.gemv =
          p.gemv_use_t ? tune.pick_gemv_t(dt, outlen, K, align, use_mpp) : tune.pick_gemv_nt(dt, K, align, use_mpp);
      p.path = GemmPlan::Path::Gemv;
      return p;
    }
  }

  // 2. Transposed-B thin-M GEMV (x @ W.t(), the lm-head / vocab path).
  if (b.trans && (dt == kHalf || dt == kBFloat16) && use_mpp && !a.trans && M >= 2 && M <= 16 && K >= 64 && N >= 16 &&
      N <= 262144) {
    const int64_t align = b.ld | a.ld | a.view.storage_offset() | b.view.storage_offset() |
        (batched ? (a.view.stride(0) | b.view.stride(0)) : 0);
    const GemvBtSpec spec = tune.pick_gemv_bt(M, N, K, /*trans_b=*/true, align);
    if (spec.valid) {
      p.gemv_bt = spec;
      p.path = GemmPlan::Path::GemvBt;
      return p;
    }
  }

  // 3. Integer GEMM (the matrix/tensor units are float-only).
  if (c10::isIntegralType(dt, /*includeBool=*/false)) {
    p.int_tile = tune.pick_int(M, N, K, dt);
    p.path = GemmPlan::Path::Int;
    return p;
  }

  // 4. float/half/bf16: matmul2d for M>=2, N>=32; else simd (pre-26.2, or narrow N).
  // matmul2d wins even at tiny K on NAX, so there is no small-K simd gate.
  const bool use_mpp_gemm = use_mpp && M >= 2 && N >= 32;
  if (!use_mpp_gemm) {
    p.simd_tile = tune.pick_simd(M, N, K);
    p.path = GemmPlan::Path::Simd;
    return p;
  }

  // 4a. Pack a thin-K column-major operand feeding a large output (see GemmTuning).
  if (!batched && (dt == kHalf || dt == kBFloat16 || dt == kFloat) && K <= tune.contig_max_k &&
      M >= tune.contig_min_mn && N >= tune.contig_min_mn && (a.trans || b.trans)) {
    p.contiguify_a = a.trans;
    p.contiguify_b = b.trans;
  }

  // 4b. Row-major thin-M decode (M in 2..4): gemv_bt streams B once and beats the
  // mostly-masked matmul tile. Larger M (where gemv_bt cliffs) stays on the tile.
  const bool packed_untrans = !a.trans && !b.trans && a.ld == K && b.ld == N;
  if (packed_untrans && (dt == kHalf || dt == kBFloat16) && M >= 2 && M <= 4 && K >= 64 && N >= 16) {
    const int64_t align = b.ld | b.view.storage_offset() | (batched ? b.view.stride(0) : 0);
    const GemvBtSpec gbt = tune.pick_gemv_bt(M, N, K, /*trans_b=*/false, align);
    if (gbt.valid) {
      p.gemv_bt = gbt;
      p.path = GemmPlan::Path::GemvBt;
      return p;
    }
  }

  // 4c. Split-K: tiny output + deep K starves matmul2d for tiles. 2-D, packed, no
  // epilogue (the reduce writes out = A@B; addmm stays on matmul2d).
  if (!batched && packed_untrans && epi == at_gemm::GemmEpilogue::None && is_splitk_regime(M, N, K, dt)) {
    const SplitKSpec sk = pick_splitk(K);
    if (sk.valid) {
      p.splitk = sk;
      p.path = GemmPlan::Path::SplitK;
      return p;
    }
  }

  // 4d. conv1x1: tall, very narrow-N projections. 2-D, packed, None epilogue.
  if (!batched && packed_untrans && epi == at_gemm::GemmEpilogue::None && is_conv_regime(M, N, K, dt)) {
    const ConvSpec cv = pick_conv(M, N);
    if (cv.valid) {
      p.conv = cv;
      p.path = GemmPlan::Path::Conv;
      return p;
    }
  }

  // 4e. matmul2d tile (bmm prefers a bigger tile; 2-D uses the stride/shape pick).
  p.mpp_tile = batched ? tune.pick_bmm(M, N, K, dt)
      : packed_untrans ? tune.pick_mpp_packed(M, N, K, dt)
                       : tune.pick_mpp(M, N, K, dt);
  p.path = GemmPlan::Path::Mpp;
  return p;
}

} // namespace

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

  // All routing/tile/contiguify heuristics live in gemm_plan; the rest of this
  // function only executes the plan it returns.
  const GemmPlan plan =
      gemm_plan(gemm_gpu_gen(), A, B, target, ra, rb, M, N, K, r, c, batched, dt, force_precise_fp32, epi);

  // Pack an operand the plan flagged (a thin column-major operand feeding a large
  // output). The copy must outlive the launch, so it is held here.
  Tensor a_packed, b_packed;
  if (plan.contiguify_a) {
    a_packed = ra.view.contiguous();
    ra = resolve_mat(a_packed, r, c);
  }
  if (plan.contiguify_b) {
    b_packed = rb.view.contiguous();
    rb = resolve_mat(b_packed, r, c);
  }
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

  const std::array<float, 2> alpha_beta = {static_cast<float>(alpha.toDouble()), static_cast<float>(beta.toDouble())};

  switch (plan.path) {
    case GemmPlan::Path::Gemv: {
      // 2-D rank-1 GEMV: bandwidth-bound, avoids the ~97%-masked GEMM tile.
      const bool m_is_one = (M == 1);
      const int64_t outlen = m_is_one ? N : M;
      const auto& mat = m_is_one ? rb : ra; // M==1: matrix is B; N==1: matrix is A
      const int64_t vec_xs = m_is_one ? A.stride(c) : B.stride(r);
      const Tensor& vmat = mat.view;
      const Tensor& vvec = m_is_one ? A : B;
      int32_t out_stride = 0;
      if (epi == at_gemm::GemmEpilogue::AlphaBeta) {
        out_stride = static_cast<int32_t>(m_is_one ? self_e.stride(c) : self_e.stride(r));
      }
      // GemvDims: {n, K, ld, xs, self_r, self_c}. gemv_t indexes self at (0,n) so the
      // addend step goes in self_c; gemv_nt indexes (row,0) so it goes in self_r.
      const std::array<int32_t, 6> gdims = {static_cast<int32_t>(outlen),
                                            static_cast<int32_t>(K),
                                            static_cast<int32_t>(mat.ld),
                                            static_cast<int32_t>(vec_xs),
                                            plan.gemv_use_t ? 0 : out_stride,
                                            plan.gemv_use_t ? out_stride : 0};
      const std::string fname = plan.gemv_use_t ? gemv_t_name(dt_str, plan.gemv.nw, plan.gemv.vec, epi)
                                                : gemv_nt_name(dt_str, plan.gemv.nw, plan.gemv.vec, epi);
      auto pso = lib.getPipelineStateForFunc(fname);
      const NSUInteger tg = static_cast<NSUInteger>(plan.gemv.nw * 32);
      const int64_t ng = plan.gemv_use_t ? ((outlen + 32 * plan.gemv.vec - 1) / (32 * plan.gemv.vec))
                                         : ((outlen + plan.gemv.nw - 1) / plan.gemv.nw);
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
      break;
    }

    case GemmPlan::Path::GemvBt: {
      // Thin-M GEMV against B (row- or column-major per gemv_bt.trans_b): streams B
      // once, no high-M cliff. The lm-head / vocab projection and the M in 2..4 decode.
      launch_gemv_bt(dt_str,
                     plan.gemv_bt,
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
      break;
    }

    case GemmPlan::Path::Int: {
      // Integer GEMM (matrix/tensor units are float-only). alpha/beta bind in the
      // accumulate type: int for <=32-bit operands, long otherwise.
      const IntTile t = plan.int_tile;
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
      break;
    }
    case GemmPlan::Path::Mpp: {
      const TensorTile t = plan.mpp_tile;
      const std::string fname = mpp_name(dt_str, t, ra.trans, rb.trans, plan.relaxed, epi, batched);
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
      break;
    }
    case GemmPlan::Path::SplitK: {
      // Tiny-output deep-K: split K across G threadgroups (out = A@B, None epilogue).
      launch_splitk(dt_str, ra.view, rb.view, target, M, N, K, plan.splitk);
      break;
    }
    case GemmPlan::Path::Conv: {
      // Tall, narrow-N projection via the 1x1-conv kernel (out = A@B, None epilogue).
      launch_conv(dt_str, ra.view, rb.view, target, M, N, K, plan.conv);
      break;
    }
    case GemmPlan::Path::Simd: {
      const SimdTile tile = plan.simd_tile;
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
      break;
    }
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
