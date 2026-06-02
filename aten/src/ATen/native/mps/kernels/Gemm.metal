// Gemm.metal - binder + explicit-instantiation site for the hand-written Metal
// GEMM kernels. This is the file the build globs and compiles (twice: once
// -std=metal3.1 -> kernels_basic.metallib, once -std=metal4.0 ->
// kernels_40.metallib). The Metal-3 families (simd, gemv) instantiate in both;
// the tensor-unit families (matmul2d) live behind #if __METAL_VERSION__ >= 400
// and only populate kernels_40.metallib.
//
// Host names encode (family, dtype, tile, flags, epilogue, batched) so the host
// dispatcher (operations/GemmMetal.mm) can build the function name at runtime.
#include <ATen/native/mps/kernels/gemm_m5_tensor.h>
#include <ATen/native/mps/kernels/gemm_simd.h>

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
// m5_tensor_gemm (Metal 4 only): mpp::tensor_ops::matmul2d on the M5 tensor unit.
// name: gemm_m5t_{dt}_{BM}_{BN}_{NSG}_{relaxed|full}_{none|ab}_{b0|b1}
// ---------------------------------------------------------------------------
#if __METAL_VERSION__ >= 400
#define MB_M5T(DT, BM, BN, NSG, RXN, RXV, EN, EV, BTN, BTV)                  \
  template [[host_name("gemm_m5t_" #DT "_" #BM "_" #BN "_" #NSG "_" #RXN "_" \
                       #EN "_" #BTN)]] kernel void                          \
  m5_tensor_gemm<DT, DT, BM, BN, NSG, RXV, GemmEpilogue::EV, BTV>(          \
      device DT*,                                                           \
      device DT*,                                                           \
      device DT*,                                                           \
      constant GemmDimsPacked&,                                            \
      device const DT*,                                                     \
      constant ::c10::metal::array<::c10::metal::opmath_t<DT>, 2>&,        \
      uint3);

#define MB_M5T_EB(DT, BM, BN, NSG, RXN, RXV)                  \
  MB_M5T(DT, BM, BN, NSG, RXN, RXV, none, None, b0, false)    \
  MB_M5T(DT, BM, BN, NSG, RXN, RXV, none, None, b1, true)     \
  MB_M5T(DT, BM, BN, NSG, RXN, RXV, ab, AlphaBeta, b0, false) \
  MB_M5T(DT, BM, BN, NSG, RXN, RXV, ab, AlphaBeta, b1, true)

// Low precision: relaxed only. fp32: relaxed (default) + full (highest precision).
#define MB_M5T_LP(DT, BM, BN, NSG) MB_M5T_EB(DT, BM, BN, NSG, relaxed, true)
#define MB_M5T_FP(DT, BM, BN, NSG)               \
  MB_M5T_EB(float, BM, BN, NSG, relaxed, true)   \
  MB_M5T_EB(float, BM, BN, NSG, full, false)

#define MB_M5T_TILES(MAC, DT) \
  MAC(DT, 16, 128, 4)         \
  MAC(DT, 32, 128, 4)         \
  MAC(DT, 32, 32, 4)          \
  MAC(DT, 16, 64, 2)          \
  MAC(DT, 32, 64, 2)          \
  MAC(DT, 48, 128, 4)         \
  MAC(DT, 64, 128, 4)         \
  MAC(DT, 128, 128, 8)        \
  MAC(DT, 64, 64, 2)

MB_M5T_TILES(MB_M5T_LP, half)
MB_M5T_TILES(MB_M5T_LP, bfloat)
MB_M5T_TILES(MB_M5T_FP, float)
#endif // __METAL_VERSION__ >= 400

} // namespace at_gemm
