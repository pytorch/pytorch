// Gemm.metal - binder + explicit-instantiation site for the hand-written Metal
// GEMM kernels. This is the file the build globs and compiles (twice: once
// -std=metal3.1 -> kernels_basic.metallib, once -std=metal4.0 ->
// kernels_40.metallib). The Metal-3 families (simd, gemv) instantiate in both;
// the tensor-unit families (matmul2d) live behind #if __METAL_VERSION__ >= 400
// and only populate kernels_40.metallib.
//
// Host names encode (family, dtype, tile, flags, epilogue, batched) so the host
// dispatcher (operations/GemmMetal.mm) can build the function name at runtime.
#include <ATen/native/mps/kernels/gemm_cgemv.h>
#include <ATen/native/mps/kernels/gemm_complex.h>
#include <ATen/native/mps/kernels/gemm_conv1x1.h>
#include <ATen/native/mps/kernels/gemm_gemv.h>
#include <ATen/native/mps/kernels/gemm_int.h>
#include <ATen/native/mps/kernels/gemm_simd.h>
#include <ATen/native/mps/kernels/gemm_splitk.h>
#include <ATen/native/mps/kernels/mpp_gemm.h>

namespace at_gemm {

// ---------------------------------------------------------------------------
// simd_gemm (Metal 3 + 4): portable simdgroup_matrix fallback.
// name: gemm_simd_{dt}_{BM}_{BN}_{BK}_{WM}_{WN}_ta{0|1}_tb{0|1}_{none|ab}_{b0|b1}
// ---------------------------------------------------------------------------
#define MB_SIMD(DT, BM, BN, BK, WM, WN, TAN, TAV, TBN, TBV, EN, EV, BTN, BTV) \
  template [[host_name("gemm_simd_" #DT "_" #BM "_" #BN "_" #BK "_" #WM "_" #WN \
                       "_ta" #TAN "_tb" #TBN "_" #EN "_" #BTN)]] kernel void   \
  simd_gemm<DT, DT, BM, BN, BK, WM, WN, TAV, TBV, GemmEpilogue::EV, BTV>(      \
      device const DT*,                                                        \
      device const DT*,                                                        \
      device DT*,                                                              \
      constant GemmDimsStrided&,                                              \
      device const DT*,                                                        \
      constant ::c10::metal::array<::c10::metal::opmath_t<DT>, 2>&,           \
      uint3,                                                                   \
      uint,                                                                    \
      uint);

// All (epilogue x batched) variants for one tile + transpose pair.
#define MB_SIMD_EB(DT, BM, BN, BK, WM, WN, TAN, TAV, TBN, TBV)        \
  MB_SIMD(DT, BM, BN, BK, WM, WN, TAN, TAV, TBN, TBV, none, None, b0, false) \
  MB_SIMD(DT, BM, BN, BK, WM, WN, TAN, TAV, TBN, TBV, none, None, b1, true)  \
  MB_SIMD(DT, BM, BN, BK, WM, WN, TAN, TAV, TBN, TBV, ab, AlphaBeta, b0, false) \
  MB_SIMD(DT, BM, BN, BK, WM, WN, TAN, TAV, TBN, TBV, ab, AlphaBeta, b1, true)

// All transpose combos for one tile.
#define MB_SIMD_TRANS(DT, BM, BN, BK, WM, WN)            \
  MB_SIMD_EB(DT, BM, BN, BK, WM, WN, 0, false, 0, false) \
  MB_SIMD_EB(DT, BM, BN, BK, WM, WN, 1, true, 0, false)  \
  MB_SIMD_EB(DT, BM, BN, BK, WM, WN, 0, false, 1, true)  \
  MB_SIMD_EB(DT, BM, BN, BK, WM, WN, 1, true, 1, true)

#define MB_SIMD_ALL(DT)                 \
  MB_SIMD_TRANS(DT, 32, 32, 16, 1, 1)   \
  MB_SIMD_TRANS(DT, 64, 64, 16, 2, 2)   \
  MB_SIMD_TRANS(DT, 128, 128, 16, 4, 4)

MB_SIMD_ALL(float)
MB_SIMD_ALL(half)
MB_SIMD_ALL(bfloat)

// ---------------------------------------------------------------------------
// int_gemm (Metal 3 + 4): register-tiled integer GEMM (no float-only matrix unit).
// name: gemm_int_{dt}_{BM}_{BN}_{BK}_{TX}_{TY}_ta{0|1}_tb{0|1}_{none|ab}_{b0|b1}
// ---------------------------------------------------------------------------
#define MB_INT(DT, BM, BN, BK, TX, TY, TAN, TAV, TBN, TBV, EN, EV, BTN, BTV) \
  template [[host_name("gemm_int_" #DT "_" #BM "_" #BN "_" #BK "_" #TX "_"   \
                       #TY "_ta" #TAN "_tb" #TBN "_" #EN "_" #BTN)]]         \
  kernel void                                                               \
  int_gemm<DT, BM, BN, BK, TX, TY, TAV, TBV, GemmEpilogue::EV, BTV>(        \
      device const DT*,                                                     \
      device const DT*,                                                     \
      device DT*,                                                           \
      constant GemmDimsStrided&,                                           \
      device const DT*,                                                     \
      constant ::c10::metal::array<::c10::metal::opmath_t<DT>, 2>&,         \
      uint3,                                                                \
      uint);

#define MB_INT_EB(DT, BM, BN, BK, TX, TY, TAN, TAV, TBN, TBV)              \
  MB_INT(DT, BM, BN, BK, TX, TY, TAN, TAV, TBN, TBV, none, None, b0, false) \
  MB_INT(DT, BM, BN, BK, TX, TY, TAN, TAV, TBN, TBV, none, None, b1, true)  \
  MB_INT(DT, BM, BN, BK, TX, TY, TAN, TAV, TBN, TBV, ab, AlphaBeta, b0, false) \
  MB_INT(DT, BM, BN, BK, TX, TY, TAN, TAV, TBN, TBV, ab, AlphaBeta, b1, true)

#define MB_INT_TRANS(DT, BM, BN, BK, TX, TY)            \
  MB_INT_EB(DT, BM, BN, BK, TX, TY, 0, false, 0, false) \
  MB_INT_EB(DT, BM, BN, BK, TX, TY, 1, true, 0, false)  \
  MB_INT_EB(DT, BM, BN, BK, TX, TY, 0, false, 1, true)  \
  MB_INT_EB(DT, BM, BN, BK, TX, TY, 1, true, 1, true)

#define MB_INT_ALL(DT)                 \
  MB_INT_TRANS(DT, 64, 64, 16, 16, 16) \
  MB_INT_TRANS(DT, 16, 64, 16, 16, 16) \
  MB_INT_TRANS(DT, 32, 64, 16, 8, 16)  \
  MB_INT_TRANS(DT, 64, 64, 8, 16, 16)  \
  MB_INT_TRANS(DT, 128, 64, 16, 16, 16)

MB_INT_ALL(char)
MB_INT_ALL(uchar)
MB_INT_ALL(short)
MB_INT_ALL(int)
MB_INT_ALL(long)

// ---------------------------------------------------------------------------
// complex deinterleave / interleave (Metal 3 + 4) for the decomposed complex GEMM.
// names: complex_split_{c2}, complex_combine_{c2}  (c2 = float2 | half2)
// ---------------------------------------------------------------------------
#define MB_CPLX(C2, R)                                                    \
  template [[host_name("complex_split_" #C2)]] kernel void                \
  complex_split<C2, R>(                                                   \
      device const C2*, device R*, device R*, constant uint&, uint);      \
  template [[host_name("complex_combine_" #C2)]] kernel void              \
  complex_combine<R, C2>(                                                 \
      device const R*,                                                    \
      device const R*,                                                    \
      device const R*,                                                    \
      device const R*,                                                    \
      device C2*,                                                         \
      constant uint&,                                                     \
      uint);

MB_CPLX(float2, float)
MB_CPLX(half2, half)

// ---------------------------------------------------------------------------
// GEMV (Metal 3 + 4): rank-1 (M==1 / N==1) bandwidth-bound kernels. VEC=1 is the
// universal, always-aligned config (still far better than a ~97%-masked GEMM tile).
// names: gemv_t_{dt}_{NW}_{VEC}_{none|ab}, gemv_nt_{dt}_{NW}_{VEC}_{red|nr}_{none|ab}
// ---------------------------------------------------------------------------
#define MB_GEMV_T(DT, NW, VEC, EN, EV)                                  \
  template [[host_name("gemv_t_" #DT "_" #NW "_" #VEC "_" #EN)]]        \
  kernel void gemv_t<DT, NW, VEC, GemmEpilogue::EV>(                    \
      device const DT*,                                                 \
      device const DT*,                                                 \
      device DT*,                                                       \
      constant GemvDims&,                                              \
      device const DT*,                                                 \
      constant ::c10::metal::array<::c10::metal::opmath_t<DT>, 2>&,     \
      uint3,                                                            \
      uint,                                                             \
      uint);
#define MB_GEMV_T_E(DT, NW, VEC) \
  MB_GEMV_T(DT, NW, VEC, none, None) MB_GEMV_T(DT, NW, VEC, ab, AlphaBeta)

#define MB_GEMV_NT(DT, NW, VEC, EN, EV)                                 \
  template [[host_name("gemv_nt_" #DT "_" #NW "_" #VEC "_" #EN)]]       \
  kernel void gemv_nt<DT, NW, VEC, GemmEpilogue::EV>(                   \
      device const DT*,                                                 \
      device const DT*,                                                 \
      device DT*,                                                       \
      constant GemvDims&,                                              \
      device const DT*,                                                 \
      constant ::c10::metal::array<::c10::metal::opmath_t<DT>, 2>&,     \
      uint3,                                                            \
      uint,                                                             \
      uint);
#define MB_GEMV_NT_E(DT, NW, VEC) \
  MB_GEMV_NT(DT, NW, VEC, none, None) MB_GEMV_NT(DT, NW, VEC, ab, AlphaBeta)

// fp32: VEC=1 only (a single fp32 column already spans a full 128-B line).
#define MB_GEMV_FLOAT(DT) \
  MB_GEMV_T_E(DT, 16, 1) MB_GEMV_T_E(DT, 32, 1) MB_GEMV_NT_E(DT, 4, 1)

// half/bf16: the full _gemv_pick VEC/NWARPS family (VEC scales with N for cache-
// line coverage). gemv_nt adds VEC 2/4 (each row read a cache line at a time).
#define MB_GEMV_LP(DT)                                                        \
  MB_GEMV_T_E(DT, 16, 1) MB_GEMV_T_E(DT, 32, 1) MB_GEMV_T_E(DT, 32, 2)        \
  MB_GEMV_T_E(DT, 16, 2) MB_GEMV_T_E(DT, 16, 4) MB_GEMV_T_E(DT, 8, 4)         \
  MB_GEMV_T_E(DT, 8, 8) MB_GEMV_T_E(DT, 16, 8)                                \
  MB_GEMV_NT_E(DT, 4, 1) MB_GEMV_NT_E(DT, 4, 2) MB_GEMV_NT_E(DT, 4, 4)

// integers: the per-width VEC from _IGEMV_*_CFG, clamped to the row-stride/offset
// alignment at dispatch (the clamp halves VEC, keeping NWARPS), so the full VEC
// ladder must exist. gemv_t uses NW=8; gemv_nt uses NW=4 (NW=8 for int32). ACC_T =
// opmath_t (int / long). The ladders cover every dtype's max VEC (8) and clamps.
#define MB_GEMV_INT(DT)                                                  \
  MB_GEMV_T_E(DT, 8, 1) MB_GEMV_T_E(DT, 8, 2) MB_GEMV_T_E(DT, 8, 4)      \
  MB_GEMV_T_E(DT, 8, 8)                                                  \
  MB_GEMV_NT_E(DT, 4, 1) MB_GEMV_NT_E(DT, 4, 2) MB_GEMV_NT_E(DT, 4, 4)   \
  MB_GEMV_NT_E(DT, 4, 8) MB_GEMV_NT_E(DT, 8, 1) MB_GEMV_NT_E(DT, 8, 2)   \
  MB_GEMV_NT_E(DT, 8, 4)

MB_GEMV_FLOAT(float)
MB_GEMV_LP(half)
MB_GEMV_LP(bfloat)
MB_GEMV_INT(char)
MB_GEMV_INT(uchar)
MB_GEMV_INT(short)
MB_GEMV_INT(int)
MB_GEMV_INT(long)

// ---------------------------------------------------------------------------
// cgemv (Metal 3 + 4): native interleaved-complex rank-1 GEMV (reads the matrix
// once vs the 4-real-GEMM decomposition). names:
//   cgemv_t_{c2}_{NW}_{none|ab}, cgemv_nt_{c2}_{NW}_{none|ab}  (c2 = float2|half2)
// ---------------------------------------------------------------------------
#define MB_CGEMV_T(C2, R, NW, EN, EV)                              \
  template [[host_name("cgemv_t_" #C2 "_" #NW "_" #EN)]]           \
  kernel void cgemv_t<C2, R, NW, GemmEpilogue::EV>(                \
      device const C2*,                                            \
      device const C2*,                                            \
      device C2*,                                                  \
      constant GemvDims&,                                         \
      device const C2*,                                            \
      constant ::c10::metal::array<float, 4>&,                     \
      uint3,                                                       \
      uint,                                                        \
      uint);
#define MB_CGEMV_NT(C2, R, NW, EN, EV)                             \
  template [[host_name("cgemv_nt_" #C2 "_" #NW "_" #EN)]]          \
  kernel void cgemv_nt<C2, R, NW, GemmEpilogue::EV>(               \
      device const C2*,                                            \
      device const C2*,                                            \
      device C2*,                                                  \
      constant GemvDims&,                                         \
      device const C2*,                                            \
      constant ::c10::metal::array<float, 4>&,                     \
      uint3,                                                       \
      uint,                                                        \
      uint);
#define MB_CGEMV(C2, R)                            \
  MB_CGEMV_T(C2, R, 8, none, None)                 \
  MB_CGEMV_T(C2, R, 8, ab, AlphaBeta)              \
  MB_CGEMV_NT(C2, R, 4, none, None)                \
  MB_CGEMV_NT(C2, R, 4, ab, AlphaBeta)

MB_CGEMV(float2, float)
MB_CGEMV(half2, half)

// ---------------------------------------------------------------------------
// m5_tensor_gemm (Metal 4 only): mpp::tensor_ops::matmul2d on the M5 tensor unit.
// Handles packed AND transposed/strided operands (the strided tensor view + the
// matmul2d transpose flags). name:
//   gemm_m5t_{dt}_{BM}_{BN}_{NSG}_ta{0|1}_tb{0|1}_{relaxed|full}_{none|ab}_{b0|b1}
// ---------------------------------------------------------------------------
#if __METAL_VERSION__ >= 400
#define MB_M5T(DT, BM, BN, NSG, TAN, TAV, TBN, TBV, RXN, RXV, EN, EV, BTN, BTV) \
  template [[host_name("gemm_m5t_" #DT "_" #BM "_" #BN "_" #NSG "_ta" #TAN      \
                       "_tb" #TBN "_" #RXN "_" #EN "_" #BTN)]] kernel void      \
  m5_tensor_gemm<DT, DT, BM, BN, NSG, TAV, TBV, RXV, GemmEpilogue::EV, BTV>(    \
      device DT*,                                                              \
      device DT*,                                                              \
      device DT*,                                                              \
      constant GemmDimsStrided&,                                              \
      device const DT*,                                                        \
      constant ::c10::metal::array<::c10::metal::opmath_t<DT>, 2>&,           \
      uint3);

// All (epilogue x batched) variants for one tile + transpose + precision combo.
#define MB_M5T_EB(DT, BM, BN, NSG, TAN, TAV, TBN, TBV, RXN, RXV)            \
  MB_M5T(DT, BM, BN, NSG, TAN, TAV, TBN, TBV, RXN, RXV, none, None, b0, false) \
  MB_M5T(DT, BM, BN, NSG, TAN, TAV, TBN, TBV, RXN, RXV, none, None, b1, true)  \
  MB_M5T(DT, BM, BN, NSG, TAN, TAV, TBN, TBV, RXN, RXV, ab, AlphaBeta, b0, false) \
  MB_M5T(DT, BM, BN, NSG, TAN, TAV, TBN, TBV, RXN, RXV, ab, AlphaBeta, b1, true)

// Low precision (half/bf16): relaxed only, all 4 transpose combos.
#define MB_M5T_LP(DT, BM, BN, NSG)                       \
  MB_M5T_EB(DT, BM, BN, NSG, 0, false, 0, false, relaxed, true) \
  MB_M5T_EB(DT, BM, BN, NSG, 1, true, 0, false, relaxed, true)  \
  MB_M5T_EB(DT, BM, BN, NSG, 0, false, 1, true, relaxed, true)  \
  MB_M5T_EB(DT, BM, BN, NSG, 1, true, 1, true, relaxed, true)

// fp32: untransposed gets relaxed (default) + full (highest precision); the
// transposed combos are relaxed-only (precise-fp32-transposed is rare and falls
// back to the fp32-exact simd kernel), which keeps the instantiation count bounded.
#define MB_M5T_FP(DT, BM, BN, NSG)                       \
  MB_M5T_EB(float, BM, BN, NSG, 0, false, 0, false, relaxed, true) \
  MB_M5T_EB(float, BM, BN, NSG, 0, false, 0, false, full, false)   \
  MB_M5T_EB(float, BM, BN, NSG, 1, true, 0, false, relaxed, true)  \
  MB_M5T_EB(float, BM, BN, NSG, 0, false, 1, true, relaxed, true)  \
  MB_M5T_EB(float, BM, BN, NSG, 1, true, 1, true, relaxed, true)

#define MB_M5T_TILES(MAC, DT) \
  MAC(DT, 16, 128, 4)         \
  MAC(DT, 32, 128, 4)         \
  MAC(DT, 32, 32, 4)          \
  MAC(DT, 16, 64, 2)          \
  MAC(DT, 32, 64, 2)          \
  MAC(DT, 48, 128, 4)         \
  MAC(DT, 64, 128, 4)         \
  MAC(DT, 128, 128, 8)        \
  MAC(DT, 64, 64, 2)          \
  MAC(DT, 64, 64, 4)

MB_M5T_TILES(MB_M5T_LP, half)
MB_M5T_TILES(MB_M5T_LP, bfloat)
MB_M5T_TILES(MB_M5T_FP, float)

// Extra autotuner-candidate tiles (untransposed only - the autotuner probes the
// packed path). These complete the metalBLAS _mpp_tensor_tile_candidates /
// _bmm_candidates union; the heuristic primary already comes from MB_M5T_TILES.
#define MB_M5T_UNTRANS_LP(DT, BM, BN, NSG) \
  MB_M5T_EB(DT, BM, BN, NSG, 0, false, 0, false, relaxed, true)
#define MB_M5T_UNTRANS_FP(DT, BM, BN, NSG)                       \
  MB_M5T_EB(DT, BM, BN, NSG, 0, false, 0, false, relaxed, true)  \
  MB_M5T_EB(DT, BM, BN, NSG, 0, false, 0, false, full, false)
#define MB_M5T_UNTRANS_TILES(MAC, DT)                       \
  MAC(DT, 128, 32, 2) MAC(DT, 256, 32, 4) MAC(DT, 32, 128, 2) \
  MAC(DT, 32, 256, 4) MAC(DT, 64, 32, 2) MAC(DT, 128, 32, 4)   \
  MAC(DT, 192, 32, 2) MAC(DT, 128, 64, 4)

MB_M5T_UNTRANS_TILES(MB_M5T_UNTRANS_LP, half)
MB_M5T_UNTRANS_TILES(MB_M5T_UNTRANS_LP, bfloat)
MB_M5T_UNTRANS_TILES(MB_M5T_UNTRANS_FP, float)

// ---------------------------------------------------------------------------
// split-K (Metal 4 only): deep-K / few-output-tile GEMM + fp32 reduction. Low
// precision only (the deep-K split regime), relaxed accum. names:
//   splitk_gemm_{dt}_{BM}_{BN}_{NSG}, splitk_reduce_{dt}
// ---------------------------------------------------------------------------
#define MB_SPLITK(DT, BM, BN, NSG)                                    \
  template [[host_name("splitk_gemm_" #DT "_" #BM "_" #BN "_" #NSG)]] \
  kernel void splitk_gemm<DT, BM, BN, NSG, true>(                     \
      device DT*,                                                     \
      device DT*,                                                     \
      device float*,                                                  \
      constant SplitKDims&,                                          \
      uint3);
#define MB_SPLITK_REDUCE(DT)                                \
  template [[host_name("splitk_reduce_" #DT)]] kernel void  \
  splitk_reduce<DT>(                                        \
      device const float*, device DT*, constant SplitKReduceDims&, uint);
#define MB_SPLITK_ALL(DT)                                            \
  MB_SPLITK(DT, 128, 32, 2) MB_SPLITK(DT, 64, 64, 2)                 \
  MB_SPLITK(DT, 32, 64, 2) MB_SPLITK_REDUCE(DT)

MB_SPLITK_ALL(half)
MB_SPLITK_ALL(bfloat)

// ---------------------------------------------------------------------------
// 1x1-conv (Metal 4 only): very-thin-N GEMM via convolution2d. Low precision only.
// KCONST (input-channel count) must be compile-time, so a set of common K values
// is precompiled; the host selects conv only when K matches one (else -> m5t).
// name: conv1x1_gemm_{dt}_{BMW}_{BNO}_{NSG}_{KCONST}  (BNO = N channels: 32 | 64)
// ---------------------------------------------------------------------------
#define MB_CONV(DT, BMW, BNO, NSG, KC)                                            \
  template [[host_name("conv1x1_gemm_" #DT "_" #BMW "_" #BNO "_" #NSG "_" #KC)]]  \
  kernel void conv1x1_gemm<DT, DT, BMW, BNO, NSG, KC>(                            \
      device DT*, device DT*, device DT*, constant ConvDims&, uint3);
#define MB_CONV_BNO(DT, BMW, NSG, KC) \
  MB_CONV(DT, BMW, 32, NSG, KC) MB_CONV(DT, BMW, 64, NSG, KC)
#define MB_CONV_TILES(DT, KC)                                            \
  MB_CONV_BNO(DT, 64, 2, KC) MB_CONV_BNO(DT, 64, 4, KC)                  \
  MB_CONV_BNO(DT, 128, 2, KC) MB_CONV_BNO(DT, 128, 4, KC)
#define MB_CONV_ALL(DT)                                                  \
  MB_CONV_TILES(DT, 512) MB_CONV_TILES(DT, 1024)                         \
  MB_CONV_TILES(DT, 2048) MB_CONV_TILES(DT, 4096)

MB_CONV_ALL(half)
MB_CONV_ALL(bfloat)
#endif // __METAL_VERSION__ >= 400

} // namespace at_gemm
