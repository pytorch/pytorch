// Parameters and constants shared by Attention.mm and the attention kernels.
#pragma once
#include <c10/metal/common.h>

// Simdgroups per prefill threadgroup, stacked along the Q block: one 8-row
// simdgroup-matrix fragment or one 16-row MPP fragment each.
C10_METAL_CONSTEXPR int kPrefillSimdgroups = 4;
C10_METAL_CONSTEXPR int kPrefillBQ = kPrefillSimdgroups * 8;
C10_METAL_CONSTEXPR int kMppBQ = kPrefillSimdgroups * 16;
C10_METAL_CONSTEXPR int kMppBK = 32;
// Keys per K block of a simdgroup-matrix tile: the wide tiles take fewer so
// fp32 fits the 32 KB threadgroup memory (padded against bank conflicts at
// 128 and 160, unpadded at 192).
constexpr int prefill_bk(int bd) {
  return bd < 128 ? 32 : bd == 128 ? 16 : 8;
}
// The 256 tile only fits two simdgroups' Q rows in fp32's 32 KB.
constexpr int prefill_simdgroups(int bd) {
  return bd == 256 ? 2 : kPrefillSimdgroups;
}

struct PrefillAttnParams {
  int B; // batch size
  int H; // number of query heads
  int D; // QK head dim; may exceed the tile width and is then chunked
  int qL; // query sequence length
  int kL; // key sequence length
  int gqa_factor; // num_q_heads / num_kv_heads
  float scale;
  int Dv; // V/O columns per slab, <= the tile width
  // Strides (B, H, L) - element stride in last dim is assumed to be 1.
  // 64-bit: a batch/head stride is a product of the other dimensions and
  // overflows 32 bits on large inputs, and the mask qL stride is multiplied
  // by an absolute query position, reaching qL * kL.
  ::c10::metal::array<int64_t, 3> Q_strides;
  ::c10::metal::array<int64_t, 3> K_strides;
  ::c10::metal::array<int64_t, 3> V_strides;
  ::c10::metal::array<int64_t, 3> O_strides;
};

struct PrefillAttnMaskParams {
  ::c10::metal::array<int64_t, 4> M_strides; // (B, H, qL, kL)
};

#ifdef __METAL__
// An additive mask enters the exp2 softmax as mask * log2(e). The product
// overflows finite sentinels such as finfo(float32).min to -inf, which would
// zero a fully masked row where the CPU math path (fp32) averages it:
// saturate instead, keeping a true -inf and NaN as they are.
METAL_FUNC float mask_log2e(float m) {
  const float m2 = m * M_LOG2E_F;
  return m2 == -INFINITY && m != -INFINITY ? -FLT_MAX : m2;
}
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>
#endif
